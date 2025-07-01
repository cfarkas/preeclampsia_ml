#!/usr/bin/env python
# coding: utf-8
###############################################################################
# 0.  GLOBAL CONSTANTS & REPRODUCIBILITY SEED
###############################################################################
SEED = 7          # <‑‑ controls NumPy, python‑random and every sklearn model
###############################################################################
import sys, subprocess, os, argparse, math, random, warnings
import numpy as np
np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)

ENV_NAME = "ml_preeclampsia"

###############################################################################
# 1.  (UNTOUCHED) ENV‑CREATE HELPERS – omitted for brevity
###############################################################################
def conda_env_exists(env_name=ENV_NAME):
    try:
        text = subprocess.run(["conda", "env", "list"],
                              capture_output=True, text=True,
                              check=True).stdout
        return any(line.startswith(env_name + " ")
                   or f"/{env_name}" in line for line in text.splitlines())
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] conda env list failed: {e}"); sys.exit(1)

def create_conda_env_if_needed(env_name=ENV_NAME):
    if conda_env_exists(env_name):
        print(f"[INFO] Env '{env_name}' already exists."); return
    print(f"[INFO] Creating env '{env_name}'…")
    subprocess.run(["conda", "create", "-y", "-n", env_name, "python=3.9"],
                   check=True)
    pkgs = ["pandas", "numpy", "scikit-learn", "matplotlib",
            "seaborn", "tqdm"]
    subprocess.run(["conda", "run", "-n", env_name, "pip", "install"] + pkgs,
                   check=True)
    print(f"[INFO] Env '{env_name}' ready.")

###############################################################################
# 2.  IMPORTS  (after seed so sklearn picks it up)
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (recall_score, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

plt.rcParams.update({
    "figure.dpi":     300,
    "font.size":      15,
    "axes.titlesize": 18,
    "axes.labelsize": 18
})

###############################################################################
# 3.  ARG‑PARSING
###############################################################################
def parse_arguments():
    p = argparse.ArgumentParser(
        description="ML pipeline for obstetric outcomes (deterministic seed).")
    p.add_argument("--install_conda", action="store_true")
    p.add_argument("--input",  type=str, required=False,
                   help="CSV/TSV data file – delimiter auto‑detected.")
    p.add_argument("--output", type=str, required=False,
                   help="Directory to write all outputs.")
    return p.parse_args()

###############################################################################
# 4.  MAIN
###############################################################################
def main():
    args = parse_arguments()

    if args.install_conda:
        create_conda_env_if_needed()
        if args.input and args.output:
            subprocess.run(["conda", "run", "-n", ENV_NAME, "python", __file__,
                            "--input", args.input, "--output", args.output],
                           check=True)
        sys.exit(0)

    if not args.input or not args.output:
        print("[ERROR] Both --input and --output are required."); sys.exit(1)
    os.makedirs(args.output, exist_ok=True)

    # ---------- smart delimiter detection ----------
    def smart_read(path):
        for d in [';', ',', '\t']:
            df = pd.read_csv(path, delimiter=d)
            if df.shape[1] > 1: return df
        raise ValueError("Cannot infer delimiter.")

    raw_df = smart_read(args.input)
    data   = raw_df.set_index('id') if 'id' in raw_df.columns else raw_df
    if data.columns[0].startswith('Unnamed'):
        data = data.drop(columns=data.columns[0])
    data.index.name = data.index.name or "index"

    ###########################################################################
    # 4‑A.  OUTCOME DEFINITION
    ###########################################################################
    all_out = ["gestational_age_delivery","newborn_weight","preeclampsia_onset",
               "delivery_type","newborn_vital_status","newborn_malformations",
               "eclampsia_hellp","iugr"]
    cont_out = ["gestational_age_delivery","newborn_weight"]
    cls_out  = [o for o in all_out if o not in cont_out]
    feat_cols = list(data.drop(columns=all_out, errors='ignore').columns)

    ###########################################################################
    # 4‑B.  CORRELATION MATRICES
    ###########################################################################
    # –– full matrix
    plt.figure(figsize=(25,19))
    sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap='seismic', vmin=-1, vmax=1, linewidths=.4, linecolor='white',
                cbar_kws={"shrink":0.5}, annot_kws={"size":12})
    plt.xticks(rotation=90, ha='right'); plt.title("Correlation Matrix", pad=20)
    plt.tight_layout(); plt.savefig(f"{args.output}/full_corr_matrix.pdf"); plt.close()

    # –– filtered matrix (|corr| ≥ 0.10 wrt any outcome)
    CORR_TH = 0.10
    corr = data.corr(numeric_only=True).abs()
    keep_feats = [c for c in feat_cols if corr.loc[c, cls_out+cont_out].max() >= CORR_TH]
    filt_cols  = keep_feats + all_out
    plt.figure(figsize=(22,17))
    sns.heatmap(data[filt_cols].corr(numeric_only=True), annot=True, fmt=".2f",
                cmap='seismic', vmin=-1, vmax=1, linewidths=.4, linecolor='white',
                cbar_kws={"shrink":0.5}, annot_kws={"size":11})
    plt.xticks(rotation=90, ha='right'); plt.title("Filtered Correlation Matrix", pad=20)
    plt.tight_layout(); plt.savefig(f"{args.output}/filtered_corr_matrix.pdf"); plt.close()

    ###########################################################################
    # 4‑C.  MODEL DICTS  (all with SEED)
    ###########################################################################
    classifiers = {
        "LogisticRegression":  LogisticRegression(max_iter=100, random_state=SEED,
                                                  solver='newton-cholesky'),
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "GaussianNB":          GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(random_state=SEED),
        "Random Forest":       RandomForestClassifier(bootstrap=False,
                                                      random_state=SEED),
        "GradientBoosting":    GradientBoostingClassifier(max_depth=5,
                                                          random_state=SEED),
        "SVM":                 SVC(probability=True, random_state=SEED),
        "MLP":                 MLPClassifier(hidden_layer_sizes=(100,),
                                             max_iter=300, random_state=SEED)
    }
    regressors = {
        "RandomForestRegressor":     RandomForestRegressor(n_estimators=100,
                                                           random_state=SEED),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100,
                                                               random_state=SEED)
    }

    meth_cls = list(classifiers.keys())
    meth_reg = list(regressors.keys())

    recall_df   = pd.DataFrame(0.0, index=all_out, columns=meth_cls)
    reg_metrics = {}
    importances = {o:{} for o in all_out}
    confmats    = {o:{} for o in cls_out}

    ###########################################################################
    # 4‑D.  TRAINING  – CLASSIFICATION
    ###########################################################################
    for o in cls_out:
        X = data[feat_cols]; y = data[o].copy()
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        Xtr,Xte,ytr,yte = train_test_split(
            X, y, test_size=0.2, random_state=SEED,
            stratify=y if len(y.unique())>1 else None)

        with open(f"{args.output}/split_log.txt","a") as f:
            f.write(f"{o},{(ytr==1).sum()},{(ytr==0).sum()},"
                    f"{(yte==1).sum()},{(yte==0).sum()}\n")

        sc = StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)

        for m in meth_cls:
            model = classifiers[m]; model.fit(Xtr,ytr); ypred=model.predict(Xte)
            recall_df.loc[o,m] = recall_score(yte, ypred, average='macro')
            cm = confusion_matrix(yte, ypred); confmats[o][m]=cm
            np.savetxt(f"{args.output}/confmat_{o}_{m.replace(' ','_')}.csv",
                       cm, fmt='%d', delimiter=';')
            try:
                imp = permutation_importance(model,Xte,yte,n_repeats=5,
                                             random_state=SEED,n_jobs=-1).importances_mean
            except Exception: imp = np.zeros(len(feat_cols))
            importances[o][m] = imp

    ###########################################################################
    # 4‑E.  TRAINING  – REGRESSION
    ###########################################################################
    for o in cont_out:
        X = data[feat_cols]; y = data[o].copy()
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        Xtr,Xte,ytr,yte = train_test_split(X,y, test_size=0.2,
                                           random_state=SEED)

        with open(f"{args.output}/split_log.txt","a") as f:
            f.write(f"{o},{len(ytr)},0,{len(yte)},0\n")

        sc = StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
        reg_metrics[o]={}
        for m in meth_reg:
            reg = regressors[m]; reg.fit(Xtr,ytr); ypred=reg.predict(Xte)
            mse  = mean_squared_error(yte, ypred); rmse = math.sqrt(mse)
            mae  = mean_absolute_error(yte, ypred); r2 = r2_score(yte, ypred)
            reg_metrics[o][m]=(mse,rmse,mae,r2)
            try:
                imp = permutation_importance(reg,Xte,yte,n_repeats=5,
                                             random_state=SEED,n_jobs=-1).importances_mean
            except Exception: imp = np.zeros(len(feat_cols))
            importances[o][m] = imp

    ###########################################################################
    # 5.  FIGURES  ------------------------------------------------------------
    ###########################################################################
    # helper
    def layout(n): return (1,n) if n<=4 else (2,3) if n<=6 else (3,3) if n<=9 else (3,4)

    # ---- pdfA  (confusion matrices) ----
    for o in cls_out:
        nr,nc=layout(len(meth_cls))
        fig,axes=plt.subplots(nr,nc,figsize=(4*nc,3.3*nr))
        axes=axes.flatten() if isinstance(axes,np.ndarray) else [axes]
        for i,m in enumerate(meth_cls):
            sns.heatmap(confmats[o][m],annot=True,fmt='g',cmap=plt.cm.Blues,ax=axes[i])
            axes[i].set_title(m,fontsize=12)
            axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("Actual")
        for ax in axes[len(meth_cls):]: ax.axis("off")
        plt.tight_layout(); plt.savefig(f"{args.output}/pdfA_{o}.pdf"); plt.close()

    # ---- pdfB  (bar plots – all methods) ----
    for o in cls_out:
        nr,nc=layout(len(meth_cls))
        figB,axesB=plt.subplots(nr,nc,figsize=(6*nc,5.5*nr))
        axesB=axesB.flatten() if isinstance(axesB,np.ndarray) else [axesB]
        for i,m in enumerate(meth_cls):
            imp=importances[o][m]; order=np.argsort(imp)[::-1]
            axesB[i].barh(np.array(feat_cols)[order],imp[order],color='steelblue')
            axesB[i].invert_yaxis(); axesB[i].tick_params(axis='y',labelsize=11)
            axesB[i].set_title(m,fontsize=12); axesB[i].set_xlabel("Mean Decrease")
        for ax in axesB[len(meth_cls):]: ax.axis("off")
        figB.subplots_adjust(left=0.38)
        plt.tight_layout(); plt.savefig(f"{args.output}/pdfB_{o}.pdf",
                                        bbox_inches='tight'); plt.close(figB)

    # ---- NEW pdfB_best_all  (one best method per outcome) ----
    # select best by SUM of positive permutation importance
    best_method = {}
    for o in all_out:
        method_dict = importances[o]
        best_m, best_sum = None, -np.inf
        for m, imp in method_dict.items():
            pos_sum = imp[imp>0].sum()
            if pos_sum > best_sum:
                best_sum, best_m = pos_sum, m
        best_method[o]=best_m

    nr,nc = layout(len(all_out))
    figBest,axesBest=plt.subplots(nr,nc,figsize=(7*nc,5.5*nr))
    axesBest=axesBest.flatten() if isinstance(axesBest,np.ndarray) else [axesBest]
    for i,o in enumerate(all_out):
        m = best_method[o]
        imp = importances[o][m]; order=np.argsort(imp)[::-1]
        axesBest[i].barh(np.array(feat_cols)[order],imp[order],color='steelblue')
        axesBest[i].invert_yaxis(); axesBest[i].tick_params(axis='y',labelsize=11)
        axesBest[i].set_title(f"{o}  –  {m}", fontsize=12)
        axesBest[i].set_xlabel("Mean Decrease")
    for ax in axesBest[len(all_out):]: ax.axis("off")
    figBest.subplots_adjust(left=0.38)
    plt.tight_layout(); plt.savefig(f"{args.output}/pdfB_best_all.pdf",
                                    bbox_inches='tight'); plt.close(figBest)

    # ---- pdfD  (unified importance heat‑maps) ----
    gmin,gmax = np.inf,-np.inf
    for d in importances.values():
        for arr in d.values():
            gmin=min(gmin,arr.min()); gmax=max(gmax,arr.max())

    for m in meth_cls+meth_reg:
        outs=[o for o in all_out if m in importances[o]]
        if not outs: continue
        mat=np.zeros((len(outs),len(feat_cols)))
        for r,o in enumerate(outs): mat[r,:]=importances[o][m]
        plt.figure(figsize=(0.9*len(feat_cols),1.0*len(outs)))
        sns.heatmap(mat,cmap='inferno',vmin=gmin,vmax=gmax,
                    xticklabels=feat_cols,yticklabels=outs,
                    cbar_kws={"shrink":0.6})
        plt.xticks(rotation=90,fontsize=11); plt.yticks(fontsize=11)
        plt.title(f"Importances – {m}")
        plt.tight_layout()
        plt.savefig(f"{args.output}/pdfD_{m.replace(' ','_')}.pdf"); plt.close()

    ###########################################################################
    # 6.  TEXT FILES & SUBSET CSVs  (unchanged logic)
    ###########################################################################
    with open(f"{args.output}/regression_metrics_summary.txt","w") as f:
        f.write("Regression Metrics (MSE | RMSE | MAE | R2)\n\n")
        for o in cont_out:
            f.write(f"{o}:\n")
            for m in meth_reg:
                mse,rmse,mae,r2 = reg_metrics[o][m]
                f.write(f"  {m:<25} MSE={mse:.2f} RMSE={rmse:.2f} "
                        f"MAE={mae:.2f} R2={r2:.2f}\n")
            f.write("\n")

    unions={"02":set(),"50":set(),"25":set()}
    def best_cls(o): return recall_df.loc[o].idxmax()
    def best_reg(o): return min(reg_metrics[o], key=lambda m: reg_metrics[o][m][0])
    for o in all_out:
        m = best_cls(o) if o in cls_out else best_reg(o)
        imp=importances[o][m]; order=np.argsort(imp)[::-1]
        mask=imp>0.02
        unions["02"].update(np.array(feat_cols)[mask] if np.any(mask)
                            else [feat_cols[order[0]]])
        unions["50"].update(np.array(feat_cols)[order[:max(1,int(len(order)*0.50))]])
        unions["25"].update(np.array(feat_cols)[order[:max(1,int(len(order)*0.25))]])
    for tag,feat_set in [("overall_subset",unions["02"]),
                         ("top50_percent",unions["50"]),
                         ("top25_percent",unions["25"])]:
        data[list(feat_set)+all_out].to_csv(f"{args.output}/subset_{tag}.csv",
                                            sep=';',index=True)

    # ---- recall heat‑map ----
    plt.figure(figsize=(1.4*len(meth_cls),1.1*len(cls_out)))
    sns.heatmap(recall_df.loc[cls_out],annot=True,fmt=".2f",cmap='inferno')
    plt.title("Recall Scores (macro)",pad=15)
    plt.xlabel("Methods"); plt.ylabel("Outcomes")
    plt.tight_layout(); plt.savefig(f"{args.output}/recall_scores_heatmap.pdf")
    plt.close()

    print(f"\n[INFO] Pipeline complete.  Outputs in: {args.output}\n")

###############################################################################
if __name__ == "__main__":
    main()

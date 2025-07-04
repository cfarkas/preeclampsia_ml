#!/usr/bin/env python
# coding: utf-8
###############################################################################
# 0.  GLOBAL SEED FOR REPRODUCIBILITY
###############################################################################
SEED = 7
import sys, subprocess, os, argparse, math, random, warnings
import numpy as np
np.random.seed(SEED); random.seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)
ENV_NAME = "ml_preeclampsia"

###############################################################################
# 1.  (conda‑env helpers unchanged – omitted for brevity)
###############################################################################
def conda_env_exists(env_name=ENV_NAME):
    try:
        txt = subprocess.run(["conda", "env", "list"],
                             capture_output=True, text=True, check=True).stdout
        return any(line.startswith(env_name + " ") or f"/{env_name}" in line
                   for line in txt.splitlines())
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] conda env list failed: {e}"); sys.exit(1)

def create_conda_env_if_needed(env_name=ENV_NAME):
    if conda_env_exists(env_name):
        print(f"[INFO] Env '{env_name}' already exists."); return
    print(f"[INFO] Creating env '{env_name}'…")
    subprocess.run(["conda", "create", "-y", "-n", env_name, "python=3.9"],
                   check=True)
    pkgs = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "tqdm"]
    subprocess.run(["conda", "run", "-n", env_name, "pip", "install"] + pkgs,
                   check=True)
    print(f"[INFO] Env '{env_name}' ready.")

###############################################################################
# 2.  IMPORTS  (after seed)
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
    "font.size":      17,     # +2 pt for easier reading
    "axes.titlesize": 20,
    "axes.labelsize": 20
})

###############################################################################
# 3.  ARG‑PARSING
###############################################################################
def parse_arguments():
    p = argparse.ArgumentParser(
        description="Deterministic ML pipeline for obstetric outcomes.")
    p.add_argument("--install_conda", action="store_true")
    p.add_argument("--input",  type=str, required=False,
                   help="CSV/TSV data file; delimiter auto‑detected.")
    p.add_argument("--output", type=str, required=False,
                   help="Directory for all outputs.")
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

    # ---------- smart delimiter ----------
    def smart_read(path):
        for d in [';', ',', '\t']:
            df = pd.read_csv(path, delimiter=d)
            if df.shape[1] > 1: return df
        raise ValueError("Cannot detect delimiter.")

    df = smart_read(args.input)
    data = df.set_index('id') if 'id' in df.columns else df
    if data.columns[0].startswith('Unnamed'):
        data = data.drop(columns=data.columns[0])
    data.index.name = data.index.name or "index"

    ###########################################################################
    # 4‑A.  OUTCOME & FEATURE LISTS
    ###########################################################################
    outcomes = ["gestational_age_delivery","newborn_weight","preeclampsia_onset",
                "delivery_type","newborn_vital_status","newborn_malformations",
                "eclampsia_hellp","iugr"]
    cont_out = ["gestational_age_delivery","newborn_weight"]
    cls_out  = [o for o in outcomes if o not in cont_out]
    feat_cols = list(data.drop(columns=outcomes, errors='ignore').columns)

    ###########################################################################
    # 4‑B.  CORRELATION MATRICES  (bigger canvases)
    ###########################################################################
    def big_corr(df, title, fname):
        plt.figure(figsize=(26,20))            # << enlarged
        sns.heatmap(df, annot=True, fmt=".2f",
                    cmap='seismic', vmin=-1, vmax=1,
                    linewidths=.4, linecolor='white',
                    cbar_kws={"shrink":0.6},
                    annot_kws={"size":13})     # << slightly larger numbers
        plt.xticks(rotation=90, ha='right', fontsize=13)
        plt.yticks(fontsize=13)
        plt.title(title, pad=25)
        plt.tight_layout()
        plt.savefig(f"{args.output}/{fname}")
        plt.close()

    big_corr(data.corr(numeric_only=True),
             "Correlation Matrix", "full_corr_matrix.pdf")

    # filtered heat‑map (|corr| ≥ 0.10)
    CORR_TH = 0.10
    corr_abs = data.corr(numeric_only=True).abs()
    keep_feat = [c for c in feat_cols if corr_abs.loc[c, outcomes].max() >= CORR_TH]
    big_corr(data[keep_feat + outcomes].corr(numeric_only=True),
             "Filtered Correlation Matrix", "filtered_corr_matrix.pdf")

    ###########################################################################
    # 4‑C.  MODEL DICTIONARIES – seeded
    ###########################################################################
    classifiers = {
        "LogReg":  LogisticRegression(max_iter=100, random_state=SEED,
                                      solver='newton-cholesky'),
        "LDA":     LinearDiscriminantAnalysis(),
        "GNB":     GaussianNB(),
        "KNN":     KNeighborsClassifier(n_neighbors=5),
        "DecTree": DecisionTreeClassifier(random_state=SEED),
        "RF":      RandomForestClassifier(bootstrap=False, random_state=SEED),
        "GradBoost": GradientBoostingClassifier(max_depth=5, random_state=SEED),
        "SVM":     SVC(probability=True, random_state=SEED),
        "MLP":     MLPClassifier(hidden_layer_sizes=(100,),
                                 max_iter=300, random_state=SEED)
    }
    regressors = {
        "RFreg":   RandomForestRegressor(n_estimators=100, random_state=SEED),
        "GBreg":   GradientBoostingRegressor(n_estimators=100, random_state=SEED)
    }
    meth_cls = list(classifiers.keys()); meth_reg = list(regressors.keys())

    recall_df = pd.DataFrame(0.0, index=outcomes, columns=meth_cls)
    reg_metrics, importances, confmats = {}, {o:{} for o in outcomes}, {o:{} for o in cls_out}

    ###########################################################################
    # 4‑D.  TRAINING  (classification)
    ###########################################################################
    for o in cls_out:
        X = data[feat_cols]; y = data[o]
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2,
                                           stratify=y if len(y.unique())>1 else None,
                                           random_state=SEED)
        with open(f"{args.output}/split_log.txt","a") as f:
            f.write(f"{o},{(ytr==1).sum()},{(ytr==0).sum()},"
                    f"{(yte==1).sum()},{(yte==0).sum()}\n")
        sc = StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)

        for m in meth_cls:
            mdl = classifiers[m]; mdl.fit(Xtr,ytr); ypred=mdl.predict(Xte)
            recall_df.loc[o,m] = recall_score(yte, ypred, average='macro')
            cm = confusion_matrix(yte, ypred); confmats[o][m]=cm
            np.savetxt(f"{args.output}/confmat_{o}_{m}.csv", cm, fmt='%d', delimiter=';')
            try:
                imp = permutation_importance(mdl,Xte,yte,n_repeats=5,
                                             random_state=SEED,n_jobs=-1).importances_mean
            except Exception: imp=np.zeros(len(feat_cols))
            importances[o][m]=imp

    ###########################################################################
    # 4‑E.  TRAINING  (regression)
    ###########################################################################
    for o in cont_out:
        X = data[feat_cols]; y = data[o]
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2,
                                           random_state=SEED)
        with open(f"{args.output}/split_log.txt","a") as f:
            f.write(f"{o},{len(ytr)},0,{len(yte)},0\n")
        sc = StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
        reg_metrics[o]={}
        for m in meth_reg:
            mdl = regressors[m]; mdl.fit(Xtr,ytr); ypred=mdl.predict(Xte)
            mse = mean_squared_error(yte, ypred); rmse=math.sqrt(mse)
            mae = mean_absolute_error(yte, ypred); r2=r2_score(yte, ypred)
            reg_metrics[o][m]=(mse,rmse,mae,r2)
            try:
                imp = permutation_importance(mdl,Xte,yte,n_repeats=5,
                                             random_state=SEED,n_jobs=-1).importances_mean
            except Exception: imp=np.zeros(len(feat_cols))
            importances[o][m]=imp

    ###########################################################################
    # 5‑A.  CONFUSION‑MATRIX GRIDS  (pdfA_*)
    ###########################################################################
    def layout(n): return (1,n) if n<=4 else (2,3) if n<=6 else (3,3) if n<=9 else (3,4)
    for o in cls_out:
        nr,nc=layout(len(meth_cls))
        fig,ax=plt.subplots(nr,nc,figsize=(4.2*nc,3.6*nr))
        ax=ax.flatten() if isinstance(ax,np.ndarray) else [ax]
        for i,m in enumerate(meth_cls):
            sns.heatmap(confmats[o][m],annot=True,fmt='g',cmap=plt.cm.Blues,ax=ax[i])
            ax[i].set_title(m,fontsize=13); ax[i].set_xlabel("Predicted"); ax[i].set_ylabel("Actual")
        for a in ax[len(meth_cls):]: a.axis("off")
        plt.tight_layout(); plt.savefig(f"{args.output}/pdfA_{o}.pdf"); plt.close()

    ###########################################################################
    # 5‑B.  PERMUTATION‑IMPORTANCE BAR PLOTS  (pdfB_* and best file)
    ###########################################################################
    # all methods per outcome
    for o in cls_out:
        nr,nc=layout(len(meth_cls))
        figB,axB=plt.subplots(nr,nc,figsize=(6.4*nc,5.8*nr))
        axB=axB.flatten() if isinstance(axB,np.ndarray) else [axB]
        for i,m in enumerate(meth_cls):
            imp=importances[o][m]; order=np.argsort(imp)[::-1]
            axB[i].barh(np.array(feat_cols)[order],imp[order],color='steelblue')
            axB[i].invert_yaxis(); axB[i].tick_params(axis='y',labelsize=11)
            axB[i].set_title(m,fontsize=13); axB[i].set_xlabel("Mean Decrease")
        for a in axB[len(meth_cls):]: a.axis("off")
        figB.subplots_adjust(left=0.38)
        plt.tight_layout(); plt.savefig(f"{args.output}/pdfB_{o}.pdf",bbox_inches='tight'); plt.close()

    # best bar per outcome → single PDF
    best_method = {}
    for o in outcomes:
        best_method[o] = max(importances[o],
                             key=lambda m: importances[o][m][importances[o][m]>0].sum())
    nr,nc=layout(len(outcomes))
    figBest,axBest=plt.subplots(nr,nc,figsize=(7.2*nc,6*nr))
    axBest=axBest.flatten() if isinstance(axBest,np.ndarray) else [axBest]
    for i,o in enumerate(outcomes):
        m=best_method[o]; imp=importances[o][m]; order=np.argsort(imp)[::-1]
        axBest[i].barh(np.array(feat_cols)[order],imp[order],color='steelblue')
        axBest[i].invert_yaxis(); axBest[i].tick_params(axis='y',labelsize=11)
        axBest[i].set_title(f"{o} – {m}",fontsize=13); axBest[i].set_xlabel("Mean Decrease")
    for a in axBest[len(outcomes):]: a.axis("off")
    figBest.subplots_adjust(left=0.38)
    plt.tight_layout(); plt.savefig(f"{args.output}/best_permutation_importances.pdf",
                                    bbox_inches='tight'); plt.close()

    ###########################################################################
    # 5‑C.  DOT‑PLOTS FOR IMPORTANCES  (single PDF 'importances.pdf')
    ###########################################################################
    gmin,gmax = np.inf,-np.inf
    for o in outcomes:
        for arr in importances[o].values():
            gmin=min(gmin,arr.min()); gmax=max(gmax,arr.max())
    norm = plt.Normalize(gmin, gmax); cmap = plt.cm.inferno

    nr,nc = layout(len(meth_cls+meth_reg))
    figDot,axDot = plt.subplots(nr,nc,figsize=(8*nc,5.5*nr),
                                sharex=False, sharey=False)
    axDot = axDot.flatten() if isinstance(axDot,np.ndarray) else [axDot]

    def dot_panel(mat, ax, title):
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r,c]
                ax.scatter(c, r, s=250*abs(val)/(gmax+1e-9)+10,
                           color=cmap(norm(val)))
        ax.set_xticks(range(len(feat_cols))); ax.set_xticklabels(feat_cols,
            rotation=90, fontsize=9)
        ax.set_yticks(range(len(outcomes))); ax.set_yticklabels(outcomes,
            fontsize=10)
        ax.set_title(title, fontsize=13)

    meth_all = meth_cls + meth_reg
    for idx,m in enumerate(meth_all):
        mat = np.zeros((len(outcomes), len(feat_cols)))
        for r,o in enumerate(outcomes):
            if m in importances[o]:
                mat[r,:] = importances[o][m]
        dot_panel(mat, axDot[idx], m)
    for a in axDot[len(meth_all):]:
        a.axis("off")
    figDot.subplots_adjust(wspace=0.35, hspace=0.55, bottom=0.25)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar_ax = figDot.add_axes([0.92, 0.25, 0.02, 0.5])
    figDot.colorbar(sm, cax=cbar_ax, label="Importance")
    plt.tight_layout(rect=[0,0,0.9,1])
    plt.savefig(f"{args.output}/importances.pdf")
    plt.close()

    ###########################################################################
    # 6.  METRIC FILES & FEATURE‑SUBSET CSVs (unchanged logic)
    ###########################################################################
    with open(f"{args.output}/regression_metrics_summary.txt","w") as f:
        f.write("Regression Metrics (MSE | RMSE | MAE | R2)\n\n")
        for o in cont_out:
            f.write(f"{o}:\n")
            for m in meth_reg:
                mse,rmse,mae,r2 = reg_metrics[o][m]
                f.write(f"  {m:<10} MSE={mse:.2f} RMSE={rmse:.2f} "
                        f"MAE={mae:.2f} R2={r2:.2f}\n")
            f.write("\n")

    unions={"02":set(),"50":set(),"25":set()}
    def best_cls(o): return recall_df.loc[o].idxmax()
    def best_reg(o): return min(reg_metrics[o], key=lambda m: reg_metrics[o][m][0])

    for o in outcomes:
        m = best_cls(o) if o in cls_out else best_reg(o)
        imp=importances[o][m]; order=np.argsort(imp)[::-1]; mask=imp>0.02
        unions["02"].update(np.array(feat_cols)[mask] if np.any(mask)
                            else [feat_cols[order[0]]])
        unions["50"].update(np.array(feat_cols)[order[:max(1,int(len(order)*.5))]])
        unions["25"].update(np.array(feat_cols)[order[:max(1,int(len(order)*.25))]])
    for tag,fs in [("overall_subset",unions["02"]),
                   ("top50_percent",unions["50"]),
                   ("top25_percent",unions["25"])]:
        data[list(fs)+outcomes].to_csv(f"{args.output}/subset_{tag}.csv",
                                       sep=';', index=True)

    # recall heat‑map
    plt.figure(figsize=(1.4*len(meth_cls),1.2*len(cls_out)))
    sns.heatmap(recall_df.loc[cls_out],annot=True,fmt=".2f",cmap='inferno')
    plt.title("Recall Scores (macro)",pad=18)
    plt.xlabel("Methods"); plt.ylabel("Outcomes")
    plt.tight_layout(); plt.savefig(f"{args.output}/recall_scores_heatmap.pdf")
    plt.close()

    print(f"\n[INFO] Pipeline complete – outputs in: {args.output}\n")

###############################################################################
if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8
###############################################################################
# 0.  ENVIRONMENT‑CREATION HELPERS
###############################################################################
import sys
import subprocess
import os
ENV_NAME = "ml_preeclampsia"

def conda_env_exists(env_name=ENV_NAME):
    try:
        out = subprocess.run(["conda", "env", "list"],
                             capture_output=True, text=True, check=True).stdout
        return any(line.startswith(env_name + " ") or f"/{env_name}" in line
                   for line in out.splitlines())
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Unable to list conda envs: {e}")
        sys.exit(1)

def create_conda_env_if_needed(env_name=ENV_NAME):
    try:
        if not conda_env_exists(env_name):
            print(f"[INFO] Creating env '{env_name}' (Python 3.9)…")
            subprocess.run(["conda", "create", "-y", "-n", env_name, "python=3.9"],
                           check=True)
            pkgs = ["pandas", "numpy", "scikit-learn", "matplotlib",
                    "seaborn", "optuna", "keras", "tensorflow", "pypi", "tqdm"]
            subprocess.run(["conda", "run", "-n", env_name, "pip", "install"] + pkgs,
                           check=True)
            subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
                            "import sklearn, sys; "
                            "print('[INFO] scikit‑learn', sklearn.__version__)"],
                           check=True)
            print(f"[INFO] Environment '{env_name}' ready.")
        else:
            print(f"[INFO] Environment '{env_name}' already exists.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Conda operation failed: {e}")
        sys.exit(1)

###############################################################################
# 1.  IMPORTS & GLOBAL MATPLOTLIB STYLE
###############################################################################
import argparse
import math
import pandas as pd
import numpy as np
import matplotlib
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
# 2.  ARGUMENT PARSING
###############################################################################
def parse_arguments():
    p = argparse.ArgumentParser(
        description="End‑to‑end ML pipeline for multiple obstetric outcomes.")
    p.add_argument("--install_conda", action="store_true",
                   help="Create/check env then re‑invoke script inside it.")
    p.add_argument("--input", type=str, required=False,
                   help="Path to CSV/TSV file (delimiter auto‑detected).")
    p.add_argument("--output", type=str, required=False,
                   help="Directory for all generated outputs.")
    return p.parse_args()

###############################################################################
# 3.  MAIN
###############################################################################
def main():
    args = parse_arguments()

    # optional environment bootstrap
    if args.install_conda:
        create_conda_env_if_needed()
        if args.input and args.output:
            subprocess.run(["conda", "run", "-n", ENV_NAME, "python", __file__,
                            "--input", args.input, "--output", args.output],
                           check=True)
        sys.exit(0)

    if not args.input or not args.output:
        print("[ERROR] Both --input and --output are required.")
        sys.exit(1)

    input_csv  = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    ###########################################################################
    # 3‑A.  DATA LOADING  (robust delimiter + optional 'id' index)
    ###########################################################################
    def smart_read_csv(path):
        for delim in [';', ',', '\t']:
            df = pd.read_csv(path, delimiter=delim)
            if df.shape[1] > 1:
                return df
        raise ValueError("Could not detect delimiter for input file.")

    raw_df = smart_read_csv(input_csv)
    if 'id' in raw_df.columns:
        data = raw_df.set_index('id')
    else:
        data = raw_df
        if data.columns[0].startswith('Unnamed'):
            data = data.drop(columns=data.columns[0])
        data.index.name = "index"

    ###########################################################################
    # 3‑B.  CORRELATION MATRIX  (larger canvas)
    ###########################################################################
    plt.figure(figsize=(22, 17))
    sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap='seismic', vmin=-1, vmax=1, linewidths=0.4,
                linecolor='white', cbar_kws={"shrink":0.5},
                annot_kws={"size": 12})
    plt.xticks(rotation=90, ha='right')
    plt.title("Correlation Matrix (Full Data)", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "full_data_corr_matrix.pdf"))
    plt.close()

    ###########################################################################
    # 3‑C.  OUTCOME LISTS & MODEL DICTIONARIES
    ###########################################################################
    all_outcomes = ["gestational_age_delivery", "newborn_weight",
                    "preeclampsia_onset", "delivery_type",
                    "newborn_vital_status", "newborn_malformations",
                    "eclampsia_hellp", "iugr"]
    continuous_outcomes     = ["gestational_age_delivery", "newborn_weight"]
    classification_outcomes = [o for o in all_outcomes if o not in continuous_outcomes]

    classifiers = {
        "LogisticRegression":     LogisticRegression(random_state=7,
                                                     solver='newton-cholesky',
                                                     max_iter=100),
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "GaussianNB":             GaussianNB(),
        "K-Nearest Neighbors":    KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":          DecisionTreeClassifier(random_state=7),
        "Random Forest":          RandomForestClassifier(bootstrap=False,
                                                         random_state=7),
        "GradientBoosting":       GradientBoostingClassifier(max_depth=5,
                                                             random_state=7),
        "SVM":                    SVC(probability=True, random_state=7),
        "MLP":                    MLPClassifier(hidden_layer_sizes=(100,),
                                                max_iter=300, random_state=7)
    }
    regressors = {
        "RandomForestRegressor":      RandomForestRegressor(n_estimators=100,
                                                            random_state=7),
        "GradientBoostingRegressor":  GradientBoostingRegressor(n_estimators=100,
                                                                random_state=7)
    }

    method_names_cls = list(classifiers.keys())
    method_names_reg = list(regressors.keys())

    # containers
    recall_df            = pd.DataFrame(0.0, index=all_outcomes,
                                        columns=method_names_cls)
    regression_metrics   = {}
    method_importances   = {o: {} for o in all_outcomes}
    method_conf_matrices = {o: {} for o in classification_outcomes}

    ###########################################################################
    # 3‑D.  CLASSIFICATION LOOP
    ###########################################################################
    for outcome_col in classification_outcomes:
        X = data.drop(columns=all_outcomes, errors='ignore')
        y = data[outcome_col].copy()

        # encode non‑numeric predictors
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=7, stratify=y if len(y.unique())>1 else None)

        # log class balance
        with open(os.path.join(output_dir, "split_log.txt"), "a") as f:
            f.write(f"{outcome_col},{(y_train==1).sum()},{(y_train==0).sum()},"
                    f"{(y_test==1).sum()},{(y_test==0).sum()}\n")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        for m in method_names_cls:
            clf = classifiers[m]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            recall_df.loc[outcome_col, m] = recall_score(
                y_test, y_pred, average='macro')

            conf = confusion_matrix(y_test, y_pred)
            method_conf_matrices[outcome_col][m] = conf
            np.savetxt(os.path.join(output_dir,
                                    f"confmat_{outcome_col}_{m.replace(' ','_')}.csv"),
                       conf, delimiter=';', fmt='%d')

            try:
                perm = permutation_importance(clf, X_test, y_test,
                                              n_repeats=5, random_state=7,
                                              n_jobs=-1).importances_mean
            except Exception:
                perm = np.zeros(X.shape[1])
            method_importances[outcome_col][m] = perm

    ###########################################################################
    # 3‑E.  REGRESSION LOOP
    ###########################################################################
    for outcome_col in continuous_outcomes:
        X = data.drop(columns=all_outcomes, errors='ignore')
        y = data[outcome_col].copy()

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=7)

        with open(os.path.join(output_dir, "split_log.txt"), "a") as f:
            f.write(f"{outcome_col},{len(y_train)},0,{len(y_test)},0\n")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        regression_metrics[outcome_col] = {}

        for m in method_names_reg:
            reg = regressors[m]
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            mse  = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(mse)
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            regression_metrics[outcome_col][m] = (mse, rmse, mae, r2)

            try:
                perm = permutation_importance(reg, X_test, y_test,
                                              n_repeats=5, random_state=7,
                                              n_jobs=-1).importances_mean
            except Exception:
                perm = np.zeros(X.shape[1])
            method_importances[outcome_col][m] = perm

    ###########################################################################
    # 4.  PLOTTING HELPERS
    ###########################################################################
    def decide_layout(n):
        if n <= 4:   return (1, n)
        if n <= 6:   return (2, 3)
        if n <= 9:   return (3, 3)
        return (3, 4)

    ###########################################################################
    # 5.  CONFUSION‑MATRIX GRIDS  (pdfA*)
    ###########################################################################
    for outcome in classification_outcomes:
        nrows, ncols = decide_layout(len(method_names_cls))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4*ncols, 3.3*nrows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        for idx, m in enumerate(method_names_cls):
            sns.heatmap(method_conf_matrices[outcome][m], annot=True, fmt='g',
                        cmap=plt.cm.Blues, ax=axes[idx])
            axes[idx].set_title(m, fontsize=12)
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("Actual")
        for ax in axes[len(method_names_cls):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pdfA_{outcome}.pdf"))
        plt.close()

    ###########################################################################
    # 6.  PERMUTATION‑IMPORTANCE BAR PLOTS (pdfB*)  --  fixed margins
    ###########################################################################
    feat_names_all = list(data.drop(columns=all_outcomes,
                                    errors='ignore').columns)

    for outcome in classification_outcomes:
        nrows, ncols = decide_layout(len(method_names_cls))
        figB, axesB = plt.subplots(nrows, ncols,
                                   figsize=(6*ncols, 5.5*nrows))
        axesB = axesB.flatten() if isinstance(axesB, np.ndarray) else [axesB]

        for idx, m in enumerate(method_names_cls):
            imps  = method_importances[outcome][m]
            order = np.argsort(imps)[::-1]
            axesB[idx].barh(np.array(feat_names_all)[order],
                            imps[order], color='steelblue')
            axesB[idx].invert_yaxis()
            axesB[idx].set_title(m, fontsize=12)
            axesB[idx].set_xlabel("Mean Decrease")
            axesB[idx].tick_params(axis='y', labelsize=11)

        for ax in axesB[len(method_names_cls):]:
            ax.axis("off")

        figB.subplots_adjust(left=0.38)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pdfB_{outcome}.pdf"),
                    bbox_inches='tight')
        plt.close(figB)

        #######################################################################
        # radial plot for same outcome (pdfC*)
        #######################################################################
        nrowsC, ncolsC = decide_layout(len(method_names_cls))
        figC, axesC = plt.subplots(nrowsC, ncolsC,
                                   figsize=(6*ncolsC, 5*nrowsC),
                                   subplot_kw=dict(polar=True))
        axesC = axesC.flatten() if isinstance(axesC, np.ndarray) else [axesC]

        n_feat = len(feat_names_all)
        base_angles = np.linspace(0, 2*math.pi, n_feat, endpoint=False)

        for idx, m in enumerate(method_names_cls):
            imps = method_importances[outcome][m]
            order = np.argsort(imps)[::-1]
            sorted_imps = imps[order]
            sorted_feats = np.array(feat_names_all)[order]
            rep_imps = np.concatenate([sorted_imps, [sorted_imps[0]]])
            angles   = np.concatenate([base_angles, [base_angles[0]]])

            axesC[idx].plot(angles, rep_imps, linewidth=2, color='darkviolet')
            axesC[idx].fill(angles, rep_imps, alpha=0.25, color='darkviolet')
            axesC[idx].set_theta_offset(math.pi/2)
            axesC[idx].set_theta_direction(-1)
            axesC[idx].set_ylim(0, max(rep_imps))
            axesC[idx].set_title(m, fontsize=12)
            axesC[idx].set_thetagrids(np.degrees(angles[:-1]),
                                      labels=sorted_feats, fontsize=6)
        for ax in axesC[len(method_names_cls):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pdfC_{outcome}.pdf"))
        plt.close(figC)

    ###########################################################################
    # 7.  RADIAL PLOTS FOR REGRESSION OUTCOMES
    ###########################################################################
    for outcome in continuous_outcomes:
        nrowsR, ncolsR = decide_layout(len(method_names_reg))
        figR, axesR = plt.subplots(nrowsR, ncolsR,
                                   figsize=(6*ncolsR, 5*nrowsR),
                                   subplot_kw=dict(polar=True))
        axesR = axesR.flatten() if isinstance(axesR, np.ndarray) else [axesR]

        n_feat = len(feat_names_all)
        base_angles = np.linspace(0, 2*math.pi, n_feat, endpoint=False)

        for idx, m in enumerate(method_names_reg):
            imps = method_importances[outcome][m]
            order = np.argsort(imps)[::-1]
            sorted_imps  = imps[order]
            sorted_feats = np.array(feat_names_all)[order]
            rep_imps = np.concatenate([sorted_imps, [sorted_imps[0]]])
            angles   = np.concatenate([base_angles, [base_angles[0]]])

            axesR[idx].plot(angles, rep_imps, linewidth=2, color='darkviolet')
            axesR[idx].fill(angles, rep_imps, alpha=0.25, color='darkviolet')
            axesR[idx].set_theta_offset(math.pi/2)
            axesR[idx].set_theta_direction(-1)
            axesR[idx].set_ylim(0, max(rep_imps))
            axesR[idx].set_title(m, fontsize=12)
            axesR[idx].set_thetagrids(np.degrees(angles[:-1]),
                                      labels=sorted_feats, fontsize=6)

        for ax in axesR[len(method_names_reg):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 f"pdfC_{outcome}_regression.pdf"))
        plt.close(figR)

    ###########################################################################
    # 8.  UNIFIED IMPORTANCE HEAT‑MAPS (pdfD*)
    ###########################################################################
    global_min = np.inf
    global_max = -np.inf
    for m in method_importances.values():
        for v in m.values():
            if len(v):
                global_min = min(global_min, v.min())
                global_max = max(global_max, v.max())

    for m in method_names_cls + method_names_reg:
        outs = [o for o in all_outcomes if m in method_importances[o]]
        if not outs:
            continue
        mat = np.zeros((len(outs), len(feat_names_all)))
        for r, o in enumerate(outs):
            imps = method_importances[o][m]
            for j, f in enumerate(feat_names_all):
                mat[r, j] = imps[j]
        nrows, ncols = mat.shape
        plt.figure(figsize=(0.9*ncols, 1.0*nrows))
        sns.heatmap(mat, cmap='inferno', vmin=global_min, vmax=global_max,
                    xticklabels=feat_names_all, yticklabels=outs,
                    cbar_kws={"shrink": 0.6})
        plt.xticks(rotation=90, fontsize=11)
        plt.yticks(fontsize=11)
        plt.title(f"Importances – {m}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pdfD_{m.replace(' ','_')}.pdf"))
        plt.close()

    ###########################################################################
    # 9.  REGRESSION METRICS SUMMARY
    ###########################################################################
    with open(os.path.join(output_dir, "regression_metrics_summary.txt"), "w") as f:
        f.write("Regression Metrics (MSE | RMSE | MAE | R2)\n\n")
        for o in continuous_outcomes:
            f.write(f"{o}:\n")
            for m in method_names_reg:
                mse, rmse, mae, r2 = regression_metrics[o][m]
                f.write(f"  {m:<25} MSE={mse:.2f}  RMSE={rmse:.2f} "
                        f" MAE={mae:.2f}  R2={r2:.2f}\n")
            f.write("\n")

    ###########################################################################
    # 10.  FEATURE‑SUBSET CSVs  (overall / top‑50 % / top‑25 %)
    ###########################################################################
    unions = {"02": set(), "50": set(), "25": set()}

    def best_cls(o): return recall_df.loc[o].idxmax()
    def best_reg(o):
        return min(regression_metrics[o], key=lambda m: regression_metrics[o][m][0])

    for o in all_outcomes:
        m = best_cls(o) if o in classification_outcomes else best_reg(o)
        imps = method_importances[o][m]
        order = np.argsort(imps)[::-1]
        unions["02"].update(np.array(feat_names_all)[imps > 0.02]
                            or [feat_names_all[order[0]]])
        unions["50"].update(np.array(feat_names_all)[order[:max(1,int(len(order)*0.50))]])
        unions["25"].update(np.array(feat_names_all)[order[:max(1,int(len(order)*0.25))]])

    for tag, feat_set in [("overall_subset", unions["02"]),
                          ("top50_percent",  unions["50"]),
                          ("top25_percent",  unions["25"])]:
        cols = list(feat_set) + all_outcomes
        data[cols].to_csv(os.path.join(output_dir, f"subset_{tag}.csv"),
                          sep=';', index=True)

    ###########################################################################
    # 11.  RECALL‑SCORE HEAT‑MAP  (Inferno)
    ###########################################################################
    plt.figure(figsize=(1.4*len(method_names_cls),
                        1.1*len(classification_outcomes)))
    sns.heatmap(recall_df.loc[classification_outcomes], annot=True, fmt=".2f",
                cmap='inferno')
    plt.title("Recall Scores (macro)", pad=15)
    plt.xlabel("Methods")
    plt.ylabel("Outcomes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recall_scores_heatmap.pdf"))
    plt.close()

    print(f"\n[INFO] Pipeline complete — outputs in '{output_dir}'.\n")

###############################################################################
# 4.  CALL MAIN
###############################################################################
if __name__ == "__main__":
    main()

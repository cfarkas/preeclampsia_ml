#!/usr/bin/env python
# coding: utf-8

###############################################################################
# Environment creation logic
###############################################################################
import sys
import subprocess
import os

ENV_NAME = "ml_preeclampsia"

def conda_env_exists(env_name=ENV_NAME):
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        env_list = result.stdout
        for line in env_list.splitlines():
            if line.startswith(env_name + " ") or f"/{env_name}" in line:
                return True
        return False
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Unable to check conda environments: {e}")
        sys.exit(1)

def create_conda_env_if_needed(env_name=ENV_NAME):
    try:
        if not conda_env_exists(env_name):
            print(f"[INFO] Creating environment '{env_name}' with Python 3.9...")
            subprocess.run(["conda", "create", "-y", "-n", env_name, "python=3.9"], check=True)

            print(f"[INFO] Installing packages in '{env_name}'...")
            packages = [
                "pandas", "numpy", "scikit-learn", "matplotlib",
                "seaborn", "optuna", "keras", "tensorflow",
                "pypi", "tqdm"
            ]
            subprocess.run(["conda", "run", "-n", env_name, "pip", "install"] + packages, check=True)

            print(f"[INFO] Checking installed tools in '{env_name}'...")
            subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
                            "import sklearn; print('[INFO] scikit-learn version:', sklearn.__version__)"],
                           check=True)

            print(f"[INFO] Environment '{env_name}' created and packages installed!")
        else:
            print(f"[INFO] Environment '{env_name}' already exists. Proceeding...")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Environment installation failed: {e}")
        sys.exit(1)

###############################################################################
# Imports for the pipeline code
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
from sklearn.metrics import (
    recall_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

# >>> GLOBAL STYLE TWEAKS — crisp text & bigger fonts everywhere
plt.rcParams["figure.dpi"]   = 300
plt.rcParams["font.size"]    = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 16

###############################################################################
# Argument Parsing
###############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a machine-learning pipeline on multiple outcome columns, producing combined PDF layouts only."
    )
    parser.add_argument(
        "--install_conda",
        action="store_true",
        help="If set, create/check the ml_preeclampsia environment, then re‑invoke script inside it."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV/TSV file (delimiter auto‑detected)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output directory for saving results."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.install_conda:
        create_conda_env_if_needed(ENV_NAME)
        if args.input and args.output:
            new_args = [__file__, "--input", args.input, "--output", args.output]
            print(f"[INFO] Re‑invoking script in '{ENV_NAME}' with: {new_args}")
            cmd = ["conda", "run", "-n", ENV_NAME, "python"] + new_args
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Re‑run in environment '{ENV_NAME}' failed: {e}")
                sys.exit(1)
        else:
            print("[INFO] Environment creation done; no pipeline run.")
        sys.exit(0)

    if not args.input or not args.output:
        print("[ERROR] You must specify --input and --output to run the pipeline.")
        sys.exit(1)

    input_csv = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    ########################
    # 1) Load Data + correlation
    ########################
    # >>> smart delimiter detection to avoid KeyError when ',' or '\t' used
    def smart_read_csv(path):
        """Return the first read that has >1 column, trying ; , then tab."""
        for delim in [';', ',', '\t']:
            df = pd.read_csv(path, delimiter=delim)
            if df.shape[1] > 1:
                return df
        raise ValueError("Unable to determine the correct delimiter of the input file.")

    raw_df = smart_read_csv(input_csv)

    # Handle optional 'id' column and stray unnamed index columns
    if 'id' in raw_df.columns:
        data = raw_df.set_index('id')
    else:
        data = raw_df
        if data.columns[0].startswith('Unnamed'):
            data = data.drop(columns=data.columns[0])
        data.index.name = "index"

    corr_matrix = data.corr(numeric_only=True)
    plt.figure(figsize=(18, 14))
    ax = sns.heatmap(
        corr_matrix, annot=True, cmap='seismic', fmt='.2f',
        annot_kws={"size":12}, linewidths=0.5, linecolor='white',
        cbar_kws={'shrink':0.5}, vmin=-1, vmax=1
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Correlation Matrix (Full Data)', fontsize=20)
    corr_path = os.path.join(output_dir, 'full_data_corr_matrix.pdf')
    plt.savefig(corr_path, bbox_inches='tight')
    plt.close()

    ########################
    # 2) Define outcomes
    ########################
    all_outcomes = [
        "gestational_age_delivery",
        "newborn_weight",
        "preeclampsia_onset",
        "delivery_type",
        "newborn_vital_status",
        "newborn_malformations",
        "eclampsia_hellp",
        "iugr"
    ]
    continuous_outcomes = ["gestational_age_delivery", "newborn_weight"]
    classification_outcomes = [o for o in all_outcomes if o not in continuous_outcomes]

    # classifiers & regressors
    classifiers = {
        "LogisticRegression": LogisticRegression(random_state=7, solver='newton-cholesky', max_iter=100),
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "GaussianNB": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=7),
        "Random Forest": RandomForestClassifier(bootstrap=False, random_state=7),
        "GradientBoosting": GradientBoostingClassifier(max_depth=5, random_state=7),
        "SVM": SVC(probability=True, random_state=7),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=7)
    }
    regressors = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=7),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=7)
    }

    method_names_cls = list(classifiers.keys())
    method_names_reg = list(regressors.keys())

    recall_df = pd.DataFrame(0.0, index=all_outcomes, columns=method_names_cls)
    regression_metrics = {}

    method_importances = {out: {} for out in all_outcomes}
    method_conf_matrices = {out: {} for out in classification_outcomes}

    ########################
    # 3) Classification
    ########################
    for outcome_col in classification_outcomes:
        X = data.drop(columns=[outcome_col])
        for o_ in all_outcomes:
            if o_ != outcome_col:
                X = X.drop(columns=o_, errors='ignore')
        y = data[outcome_col].copy()

        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=7, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=7
            )

        # >>> LOG CLASS BALANCE
        train_pos = int(np.sum(y_train == 1))
        train_neg = len(y_train) - train_pos
        test_pos  = int(np.sum(y_test == 1))
        test_neg  = len(y_test) - test_pos
        with open(os.path.join(output_dir, "split_log.txt"), "a") as lf:
            lf.write(f"{outcome_col},{train_pos},{train_neg},{test_pos},{test_neg}\n")

        sc_ = StandardScaler()
        X_train = sc_.fit_transform(X_train)
        X_test = sc_.transform(X_test)
        feat_names = np.array(X.columns)

        for m_ in method_names_cls:
            clf = classifiers[m_]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            recall_df.loc[outcome_col, m_] = recall_score(y_test, y_pred, average='macro')

            conf_ = confusion_matrix(y_test, y_pred)
            method_conf_matrices[outcome_col][m_] = conf_

            # >>> SAVE CONFUSION MATRIX
            np.savetxt(
                os.path.join(output_dir, f"confmat_{outcome_col}_{m_.replace(' ', '_')}.csv"),
                conf_, delimiter=';', fmt='%d')

            try:
                perm_res = permutation_importance(
                    clf, X_test, y_test, n_repeats=5, random_state=7, n_jobs=-1
                )
                importances = perm_res.importances_mean
            except Exception:
                importances = np.zeros(len(feat_names))

            method_importances[outcome_col][m_] = importances

    ########################
    # 4) Regression
    ########################
    for outcome_col in continuous_outcomes:
        X = data.drop(columns=[outcome_col])
        for o_ in all_outcomes:
            if o_ != outcome_col:
                X = X.drop(columns=o_, errors='ignore')
        y = data[outcome_col].copy()

        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=7
        )

        # >>> LOG SPLIT SIZE (no class concept here)
        with open(os.path.join(output_dir, "split_log.txt"), "a") as lf:
            lf.write(f"{outcome_col},{len(y_train)},0,{len(y_test)},0\n")

        sc_ = StandardScaler()
        X_train = sc_.fit_transform(X_train)
        X_test = sc_.transform(X_test)
        feat_names = np.array(X.columns)

        regression_metrics[outcome_col] = {}

        for r_ in method_names_reg:
            model_ = regressors[r_]
            model_.fit(X_train, y_train)
            y_pred = model_.predict(X_test)

            mse_val = mean_squared_error(y_test, y_pred)
            rmse_val = math.sqrt(mse_val)
            mae_val = mean_absolute_error(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)
            regression_metrics[outcome_col][r_] = (mse_val, rmse_val, mae_val, r2_val)

            try:
                perm_res = permutation_importance(
                    model_, X_test, y_test, n_repeats=5, random_state=7, n_jobs=-1
                )
                importances = perm_res.importances_mean
            except Exception:
                importances = np.zeros(len(feat_names))

            method_importances[outcome_col][r_] = importances

    ########################
    # 5) Confusion matrix plots
    ########################
    def decide_layout(n):
        if n <= 4:
            return (1, n)
        elif n <= 6:
            return (2, 3)
        elif n <= 9:
            return (3, 3)
        else:
            return (3, 4)

    for outcome_col in classification_outcomes:
        pdfA_path = os.path.join(output_dir, f"pdfA_{outcome_col}.pdf")
        n_methods = len(method_names_cls)
        nrows, ncols = decide_layout(n_methods)
        figA, axesA = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
        axesA = axesA.flatten() if nrows*ncols > 1 else [axesA]

        for idx, m_ in enumerate(method_names_cls):
            conf_ = method_conf_matrices[outcome_col][m_]
            sns.heatmap(conf_, annot=True, fmt='g', ax=axesA[idx], cmap=plt.cm.Blues)
            axesA[idx].set_title(m_, fontsize=10)
            axesA[idx].set_xlabel("Predicted")
            axesA[idx].set_ylabel("Actual")

        for ax in axesA[len(method_names_cls):]:
            ax.axis("off")

        plt.tight_layout()
        figA.suptitle(f"Combined Confusion Matrices - {outcome_col}")
        plt.savefig(pdfA_path)
        plt.close(figA)

    ########################
    # 6) Feature-importance bar & radial plots (classification)
    ########################
    for outcome_col in classification_outcomes:
        pdfB_path = os.path.join(output_dir, f"pdfB_{outcome_col}.pdf")
        n_methods = len(method_names_cls)
        nrowsB, ncolsB = decide_layout(n_methods)
        figB, axesB = plt.subplots(nrows=nrowsB, ncols=ncolsB, figsize=(4.5*ncolsB, 4.5*nrowsB))
        axesB = axesB.flatten() if nrowsB*ncolsB > 1 else [axesB]

        Xtemp = data.drop(columns=all_outcomes, errors='ignore')
        feat_names = np.array(Xtemp.columns)

        for idx, m_ in enumerate(method_names_cls):
            imps = method_importances[outcome_col][m_]
            sorted_idx = np.argsort(imps)[::-1]
            axesB[idx].barh(feat_names[sorted_idx], imps[sorted_idx], color='steelblue')
            axesB[idx].invert_yaxis()
            axesB[idx].set_title(m_, fontsize=10)
            axesB[idx].set_xlabel("Mean Decrease")

        for ax in axesB[len(method_names_cls):]:
            ax.axis("off")

        figB.suptitle(f"Permutation Importances - {outcome_col}")
        plt.tight_layout()
        plt.savefig(pdfB_path)
        plt.close(figB)

        # radial plots
        pdfC_path = os.path.join(output_dir, f"pdfC_{outcome_col}.pdf")
        nrowsC, ncolsC = decide_layout(n_methods)
        figC, axesC = plt.subplots(nrows=nrowsC, ncols=ncolsC,
                                   figsize=(6*ncolsC, 5*nrowsC),
                                   subplot_kw=dict(polar=True))
        axesC = axesC.flatten() if nrowsC*ncolsC > 1 else [axesC]

        n_feat = len(feat_names)
        base_angles = np.linspace(0, 2*math.pi, n_feat, endpoint=False).tolist()

        for idx, m_ in enumerate(method_names_cls):
            imps = method_importances[outcome_col][m_]
            sorted_idx = np.argsort(imps)[::-1]
            sorted_imps = imps[sorted_idx]
            sorted_feats = feat_names[sorted_idx]
            rep_imps = np.concatenate((sorted_imps, [sorted_imps[0]]))
            angles = base_angles + base_angles[:1]

            axesC[idx].plot(angles, rep_imps, linewidth=2, color='darkviolet')
            axesC[idx].fill(angles, rep_imps, alpha=0.25, color='darkviolet')
            axesC[idx].set_theta_offset(math.pi/2)
            axesC[idx].set_theta_direction(-1)
            axesC[idx].set_ylim(0, max(rep_imps))
            axesC[idx].set_title(m_, fontsize=10)
            axesC[idx].set_thetagrids(np.degrees(angles[:-1]), labels=sorted_feats, fontsize=6)

        for ax in axesC[len(method_names_cls):]:
            ax.axis("off")

        figC.suptitle(f"Radial Importances - {outcome_col}")
        plt.tight_layout()
        plt.savefig(pdfC_path)
        plt.close(figC)

    ########################
    # 7) Radial plots for regression
    ########################
    for outcome_col in continuous_outcomes:
        pdfC_reg_path = os.path.join(output_dir, f"pdfC_{outcome_col}_regression.pdf")
        reg_method_names = list(regressors.keys())
        nrowsR, ncolsR = decide_layout(len(reg_method_names))
        figR, axesR = plt.subplots(nrows=nrowsR, ncols=ncolsR,
                                   figsize=(6*ncolsR, 5*nrowsR),
                                   subplot_kw=dict(polar=True))
        axesR = axesR.flatten() if nrowsR*ncolsR > 1 else [axesR]

        XtempR = data.drop(columns=all_outcomes, errors='ignore')
        feat_names_R = np.array(XtempR.columns)
        n_featR = len(feat_names_R)
        base_anglesR = np.linspace(0, 2*math.pi, n_featR, endpoint=False).tolist()

        for idx, r_ in enumerate(reg_method_names):
            imps = method_importances[outcome_col][r_]
            sorted_idx = np.argsort(imps)[::-1]
            sorted_imps = imps[sorted_idx]
            sorted_feats = feat_names_R[sorted_idx]
            rep_imps = np.concatenate((sorted_imps, [sorted_imps[0]]))
            anglesR = base_anglesR + base_anglesR[:1]

            axesR[idx].plot(anglesR, rep_imps, linewidth=2, color='darkviolet')
            axesR[idx].fill(anglesR, rep_imps, alpha=0.25, color='darkviolet')
            axesR[idx].set_theta_offset(math.pi/2)
            axesR[idx].set_theta_direction(-1)
            axesR[idx].set_ylim(0, max(rep_imps))
            axesR[idx].set_title(r_, fontsize=10)
            axesR[idx].set_thetagrids(np.degrees(anglesR[:-1]), labels=sorted_feats, fontsize=6)

        for ax in axesR[len(reg_method_names):]:
            ax.axis("off")

        figR.suptitle(f"Radial Importances - {outcome_col}")
        plt.tight_layout()
        plt.savefig(pdfC_reg_path)
        plt.close(figR)

    ########################
    # 8) Unified importances heat‑maps (pdfD)
    ########################
    all_methods = method_names_cls + method_names_reg
    global_min, global_max = np.inf, -np.inf
    for m_ in all_methods:
        for o_ in all_outcomes:
            if m_ in method_importances[o_]:
                arr = method_importances[o_][m_]
                if len(arr):
                    global_min = min(global_min, np.min(arr))
                    global_max = max(global_max, np.max(arr))

    all_feats = data.drop(columns=all_outcomes, errors='ignore').columns.tolist()
    for m_ in all_methods:
        outs_used = [o_ for o_ in all_outcomes if m_ in method_importances[o_]]
        if not outs_used:
            continue
        mat = np.zeros((len(outs_used), len(all_feats)))
        for row, o_ in enumerate(outs_used):
            Xtmp = data.drop(columns=all_outcomes, errors='ignore')
            feats_o = list(Xtmp.columns)
            imps_o = method_importances[o_][m_]
            for j, feat in enumerate(feats_o):
                if feat in all_feats:
                    mat[row, all_feats.index(feat)] = imps_o[j]

        pdfD_path = os.path.join(output_dir, f"pdfD_{m_.replace(' ', '_')}.pdf")
        plt.figure(figsize=(1.2*len(all_feats), 1.2*len(outs_used)))
        sns.heatmap(mat, cmap='inferno', vmin=global_min, vmax=global_max,
                    xticklabels=all_feats, yticklabels=outs_used)
        plt.xticks(rotation=90)
        plt.title(f"Inferno Heatmap - {m_} (Importances)")
        plt.xlabel("Features")
        plt.ylabel("Outcomes")
        plt.tight_layout()
        plt.savefig(pdfD_path, bbox_inches='tight')
        plt.close()

    ########################
    # 9) Save regression metrics
    ########################
    with open(os.path.join(output_dir, "regression_metrics_summary.txt"), "w") as f:
        f.write("Regression Metrics (MSE, RMSE, MAE, R2)\n\n")
        for out_ in continuous_outcomes:
            f.write(f"Outcome: {out_}\n")
            for r_ in method_names_reg:
                mse_val, rmse_val, mae_val, r2_val = regression_metrics[out_][r_]
                f.write(f"  {r_}: MSE={mse_val:.2f}, RMSE={rmse_val:.2f}, "
                        f"MAE={mae_val:.2f}, R2={r2_val:.2f}\n")
            f.write("\n")

    ########################
    # 10) Generate best‑feature subset CSV
    ########################
    def best_method_cls(outcome):
        return recall_df.loc[outcome].idxmax()

    def best_method_reg(outcome):
        return min(regression_metrics[outcome], key=lambda m: regression_metrics[outcome][m][0])

    best_feats = set()
    cutoff = 0.02
    for o_ in classification_outcomes:
        m_ = best_method_cls(o_)
        imps = method_importances[o_][m_]
        Xtmp = data.drop(columns=all_outcomes, errors='ignore')
        feats = list(Xtmp.columns)
        idxs = np.where(imps > cutoff)[0] if np.any(imps > cutoff) else [int(np.argmax(imps))]
        best_feats.update([feats[i] for i in idxs])

    for o_ in continuous_outcomes:
        m_ = best_method_reg(o_)
        imps = method_importances[o_][m_]
        Xtmp = data.drop(columns=all_outcomes, errors='ignore')
        feats = list(Xtmp.columns)
        idxs = np.where(imps > cutoff)[0] if np.any(imps > cutoff) else [int(np.argmax(imps))]
        best_feats.update([feats[i] for i in idxs])

    subset_cols = list(best_feats) + all_outcomes
    data[subset_cols].to_csv(os.path.join(output_dir, "best_features_overall_subset.csv"),
                             sep=';', index=True)

    ########################
    # 11) Recall heat‑map
    ########################
    plt.figure(figsize=(1.5*len(method_names_cls), 1.2*len(classification_outcomes)))
    sns.heatmap(recall_df.loc[classification_outcomes], annot=True, cmap='cividis', fmt=".2f")
    plt.title("Recall Scores Heat‑Map (Classification Outcomes × Methods)")
    plt.xlabel("Methods")
    plt.ylabel("Outcomes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recall_scores_heatmap.pdf"))
    plt.close()

    print("\n[INFO] Pipeline complete.\n"
          f"      All outputs saved to: {output_dir}\n")

if __name__ == "__main__":
    main()

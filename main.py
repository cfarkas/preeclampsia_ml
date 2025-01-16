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
        help="If set, create/check the ml_preeclampsia environment, then re-invoke script inside it."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV file (with ';' delimiter and 'id' index col)."
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
            print(f"[INFO] Re-invoking script in '{ENV_NAME}' with: {new_args}")
            cmd = ["conda", "run", "-n", ENV_NAME, "python"] + new_args
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Re-run in environment '{ENV_NAME}' failed: {e}")
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
    data = pd.read_csv(input_csv, delimiter=';', index_col='id')

    corr_matrix = data.corr()
    plt.figure(figsize=(30, 24))
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
        "LogisticRegression": LogisticRegression(random_state=42, solver='newton-cholesky', max_iter=100),
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "GaussianNB": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(bootstrap=False, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(max_depth=5, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }
    regressors = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    method_names_cls = list(classifiers.keys())
    method_names_reg = list(regressors.keys())

    # Store classification recall, regression metrics
    recall_df = pd.DataFrame(0.0, index=all_outcomes, columns=method_names_cls)
    regression_metrics = {}

    method_importances = {out: {} for out in all_outcomes}
    method_conf_matrices = {out: {} for out in classification_outcomes}

    ########################
    # 3) Classification
    ########################
    for outcome_col in classification_outcomes:
        X = data.drop(columns=[outcome_col])
        others = [o for o in all_outcomes if o != outcome_col]
        for o_ in others:
            if o_ in X.columns:
                X = X.drop(columns=[o_])
        y = data[outcome_col].copy()

        # encode
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        sc_ = StandardScaler()
        X_train = sc_.fit_transform(X_train)
        X_test = sc_.transform(X_test)
        feat_names = np.array(X.columns)

        for m_ in method_names_cls:
            clf = classifiers[m_]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            rec_ = recall_score(y_test, y_pred, average='macro')
            recall_df.loc[outcome_col, m_] = rec_

            conf_ = confusion_matrix(y_test, y_pred)
            method_conf_matrices[outcome_col][m_] = conf_

            try:
                perm_res = permutation_importance(
                    clf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
                )
                importances = perm_res.importances_mean
            except:
                importances = np.zeros(len(feat_names))

            method_importances[outcome_col][m_] = importances

    ########################
    # 4) Regression
    ########################
    for outcome_col in continuous_outcomes:
        X = data.drop(columns=[outcome_col])
        others = [o for o in all_outcomes if o != outcome_col]
        for o_ in others:
            if o_ in X.columns:
                X = X.drop(columns=[o_])
        y = data[outcome_col].copy()

        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
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
                    model_, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
                )
                importances = perm_res.importances_mean
            except:
                importances = np.zeros(len(feat_names))

            method_importances[outcome_col][r_] = importances

    ########################
    # 5) Confusion matrix plots with plasma
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
        figA, axesA = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
        if nrows*ncols == 1:
            axesA = [axesA]
        else:
            axesA = axesA.flatten()

        # CHANGED: Instead of 'plasma', use white->darkblue (plt.cm.Blues)
        # plasma_cmap = plt.get_cmap("plasma")
        white_to_darkblue = plt.cm.Blues  # from white to dark blue

        i_ = 0
        for m_ in method_names_cls:
            conf_ = method_conf_matrices[outcome_col][m_]
            ax_ = axesA[i_]
            i_ += 1
            # sns.heatmap(conf_, annot=True, fmt='g', ax=ax_, cmap=plasma_cmap)
            sns.heatmap(conf_, annot=True, fmt='g', ax=ax_, cmap=white_to_darkblue)  # changed
            ax_.set_title(m_, fontsize=10)
            ax_.set_xlabel("Predicted")
            ax_.set_ylabel("Actual")

        for j in range(i_, nrows*ncols):
            axesA[j].axis("off")

        plt.tight_layout()
        figA.suptitle(f"Combined Confusion Matrices - {outcome_col}")
        plt.savefig(pdfA_path)
        plt.close(figA)

    ########################
    # 6) Bar plots, radial, etc. for classification
    ########################
    for outcome_col in classification_outcomes:
        pdfB_path = os.path.join(output_dir, f"pdfB_{outcome_col}.pdf")
        n_methods = len(method_names_cls)
        nrowsB, ncolsB = decide_layout(n_methods)
        figB, axesB = plt.subplots(nrows=nrowsB, ncols=ncolsB, figsize=(6*ncolsB, 6*nrowsB))
        if nrowsB*ncolsB == 1:
            axesB = [axesB]
        else:
            axesB = axesB.flatten()

        Xtemp = data.drop(columns=[outcome_col])
        for o_ in all_outcomes:
            if o_ != outcome_col and o_ in Xtemp.columns:
                Xtemp = Xtemp.drop(columns=[o_])
        feat_names = np.array(Xtemp.columns)

        i_ = 0
        for m_ in method_names_cls:
            imps = method_importances[outcome_col][m_]
            sorted_idx = np.argsort(imps)[::-1]
            ax_ = axesB[i_]
            i_ += 1
            bar_labels = feat_names[sorted_idx]
            ax_.barh(bar_labels, imps[sorted_idx], color='steelblue')
            ax_.invert_yaxis()
            ax_.set_title(m_, fontsize=10)
            ax_.set_xlabel("Mean Decrease")

        for j in range(i_, nrowsB*ncolsB):
            axesB[j].axis("off")

        figB.suptitle(f"Permutation Importances - {outcome_col}")
        plt.tight_layout()
        plt.savefig(pdfB_path)
        plt.close(figB)

        # radial
        pdfC_path = os.path.join(output_dir, f"pdfC_{outcome_col}.pdf")
        nrowsC, ncolsC = decide_layout(n_methods)
        figC, axesC = plt.subplots(nrows=nrowsC, ncols=ncolsC,
                                   figsize=(6*ncolsC, 5*nrowsC),
                                   subplot_kw=dict(polar=True))
        if nrowsC*ncolsC == 1:
            axesC = [axesC]
        else:
            axesC = axesC.flatten()

        Xtemp2 = data.drop(columns=[outcome_col])
        for o_ in all_outcomes:
            if o_ != outcome_col and o_ in Xtemp2.columns:
                Xtemp2 = Xtemp2.drop(columns=[o_])
        feat_names_for_radial = np.array(Xtemp2.columns)
        n_feat = len(feat_names_for_radial)
        base_angles = np.linspace(0, 2*math.pi, n_feat, endpoint=False).tolist()

        i_ = 0
        for m_ in method_names_cls:
            imps = method_importances[outcome_col][m_]
            sorted_idx = np.argsort(imps)[::-1]
            sorted_imps = imps[sorted_idx]
            sorted_feats = feat_names_for_radial[sorted_idx]

            rep_imps = np.concatenate((sorted_imps, [sorted_imps[0]]))
            angles = base_angles[:]
            angles += angles[:1]

            ax_ = axesC[i_]
            i_ += 1
            # CHANGED: color='darkviolet' instead of 'gray'
            ax_.plot(angles, rep_imps, linewidth=2, linestyle='solid', color='darkviolet')
            ax_.fill(angles, rep_imps, alpha=0.25, color='darkviolet')
            ax_.set_theta_offset(math.pi/2)
            ax_.set_theta_direction(-1)
            ax_.set_ylim(0, max(0, max(rep_imps)))
            ax_.set_title(m_, fontsize=10)

            degs = np.degrees(angles[:-1])
            ax_.set_thetagrids(degs, labels=sorted_feats, fontsize=6)

        for j in range(i_, nrowsC*ncolsC):
            axesC[j].axis("off")

        figC.suptitle(f"Radial Importances - {outcome_col}")
        plt.tight_layout()
        plt.savefig(pdfC_path)
        plt.close(figC)

    ########################
    # 7) Radial for regression
    ########################
    for outcome_col in continuous_outcomes:
        reg_method_names = list(regressors.keys())
        pdfC_reg_path = os.path.join(output_dir, f"pdfC_{outcome_col}_regression.pdf")

        nrowsR, ncolsR = decide_layout(len(reg_method_names))
        figCreg, axesCreg = plt.subplots(nrows=nrowsR, ncols=ncolsR,
                                         figsize=(6*ncolsR, 5*nrowsR),
                                         subplot_kw=dict(polar=True))
        if nrowsR*ncolsR == 1:
            axesCreg = [axesCreg]
        else:
            axesCreg = axesCreg.flatten()

        XtempR = data.drop(columns=[outcome_col])
        for o_ in all_outcomes:
            if o_ != outcome_col and o_ in XtempR.columns:
                XtempR = XtempR.drop(columns=[o_])
        feat_names_for_radial = np.array(XtempR.columns)
        n_feat = len(feat_names_for_radial)
        base_angles_r = np.linspace(0, 2*math.pi, n_feat, endpoint=False).tolist()

        i_ = 0
        for r_ in reg_method_names:
            imps = method_importances[outcome_col][r_]
            sorted_idx = np.argsort(imps)[::-1]
            sorted_imps = imps[sorted_idx]
            sorted_feats = feat_names_for_radial[sorted_idx]

            rep_imps = np.concatenate((sorted_imps, [sorted_imps[0]]))
            angles_r = base_angles_r[:]
            angles_r += angles_r[:1]

            ax_ = axesCreg[i_]
            i_ += 1
            # CHANGED: color='darkviolet' instead of 'gray'
            ax_.plot(angles_r, rep_imps, linewidth=2, linestyle='solid', color='darkviolet')
            ax_.fill(angles_r, rep_imps, alpha=0.25, color='darkviolet')
            ax_.set_theta_offset(math.pi/2)
            ax_.set_theta_direction(-1)
            ax_.set_ylim(0, max(0, max(rep_imps)))
            ax_.set_title(r_, fontsize=10)

            degs = np.degrees(angles_r[:-1])
            ax_.set_thetagrids(degs, labels=sorted_feats, fontsize=6)

        for j in range(i_, nrowsR*ncolsR):
            axesCreg[j].axis("off")

        figCreg.suptitle(f"Radial Importances - {outcome_col}")
        plt.tight_layout()
        plt.savefig(pdfC_reg_path)
        plt.close(figCreg)

    ########################
    # 8) pdfD => unify scale + actual names, use bbox_inches
    ########################
    global_minD = float('inf')
    global_maxD = float('-inf')
    all_methods = list(classifiers.keys()) + list(regressors.keys())

    for method_name in all_methods:
        for out_ in all_outcomes:
            if method_name in method_importances[out_]:
                arr_ = method_importances[out_][method_name]
                if len(arr_):
                    arr_min = np.min(arr_)
                    arr_max = np.max(arr_)
                    if arr_min < global_minD:
                        global_minD = arr_min
                    if arr_max > global_maxD:
                        global_maxD = arr_max

    all_feats = data.drop(columns=all_outcomes, errors='ignore').columns.tolist()
    n_all_feats = len(all_feats)

    for method_name in all_methods:
        used_outs = []
        for out_ in all_outcomes:
            if method_name in method_importances[out_]:
                used_outs.append(out_)

        if not used_outs:
            continue

        mat = np.zeros((len(used_outs), n_all_feats))
        row_labels = used_outs

        for i, out_ in enumerate(used_outs):
            XtempD = data.drop(columns=[out_], errors='ignore')
            for xo in all_outcomes:
                if xo != out_ and xo in XtempD.columns:
                    XtempD = XtempD.drop(columns=[xo], errors='ignore')
            feats_for_out = list(XtempD.columns)
            imps_arr = method_importances[out_][method_name]

            for j, feat_j in enumerate(feats_for_out):
                if feat_j in all_feats:
                    col_idx = all_feats.index(feat_j)
                    mat[i, col_idx] = imps_arr[j]

        pdfD_path = os.path.join(output_dir, f"pdfD_{method_name}.pdf")
        plt.figure(figsize=(1.2*n_all_feats, 1.2*len(used_outs)))
        sns.heatmap(
            mat, annot=False, cmap='inferno',
            xticklabels=all_feats, yticklabels=row_labels,
            vmin=global_minD, vmax=global_maxD
        )
        plt.xticks(rotation=90)
        plt.title(f"Inferno Heatmap - {method_name} (Importances) [Unif. Scale]")
        plt.xlabel("Features")
        plt.ylabel("Outcomes")
        plt.tight_layout()
        plt.savefig(pdfD_path, bbox_inches='tight')
        plt.close()

    ########################
    # 9) Output regression metrics
    ########################
    reg_metrics_file = os.path.join(output_dir, "regression_metrics_summary.txt")
    with open(reg_metrics_file, "w") as f:
        f.write("Regression Metrics (MSE, RMSE, MAE, R2)\n\n")
        for out_ in continuous_outcomes:
            f.write(f"Outcome: {out_}\n")
            if out_ not in regression_metrics:
                continue
            for r_ in method_names_reg:
                if r_ in regression_metrics[out_]:
                    mse_val, rmse_val, mae_val, r2_val = regression_metrics[out_][r_]
                    f.write(f"  {r_}: MSE={mse_val:.2f}, RMSE={rmse_val:.2f}, MAE={mae_val:.2f}, R2={r2_val:.2f}\n")
            f.write("\n")

    ########################
    # Subset CSV: pick best method for each outcome, get top features, unify
    ########################
    def pick_best_method_classification(outcome):
        best_m = None
        best_score = -999
        for m_ in method_names_cls:
            val = recall_df.loc[outcome, m_]
            if val > best_score:
                best_score = val
                best_m = m_
        return best_m

    def pick_best_method_regression(outcome):
        best_m = None
        best_mse = float('inf')
        if outcome in regression_metrics:
            for r_ in method_names_reg:
                if r_ in regression_metrics[outcome]:
                    (mse_, rmse_, mae_, r2_) = regression_metrics[outcome][r_]
                    if mse_ < best_mse:
                        best_mse = mse_
                        best_m = r_
        return best_m

    top_cutoff = 0.02
    best_feat_union = set()

    for out_ in classification_outcomes:
        bm_ = pick_best_method_classification(out_)
        if bm_:
            Xtemp_ = data.drop(columns=[out_], errors='ignore')
            for xo in all_outcomes:
                if xo != out_ and xo in Xtemp_.columns:
                    Xtemp_ = Xtemp_.drop(columns=[xo], errors='ignore')
            feats_ = list(Xtemp_.columns)
            imps_ = method_importances[out_][bm_]
            mask = (imps_ > top_cutoff)
            if not np.any(mask):
                top_idx = [np.argmax(imps_)]
            else:
                top_idx = np.where(mask)[0]
            for idx_ in top_idx:
                best_feat_union.add(feats_[idx_])

    for out_ in continuous_outcomes:
        bm_ = pick_best_method_regression(out_)
        if bm_:
            Xtemp_ = data.drop(columns=[out_], errors='ignore')
            for xo in all_outcomes:
                if xo != out_ and xo in Xtemp_.columns:
                    Xtemp_ = Xtemp_.drop(columns=[xo], errors='ignore')
            feats_ = list(Xtemp_.columns)
            imps_ = method_importances[out_][bm_]
            mask = (imps_ > top_cutoff)
            if not np.any(mask):
                top_idx = [np.argmax(imps_)]
            else:
                top_idx = np.where(mask)[0]
            for idx_ in top_idx:
                best_feat_union.add(feats_[idx_])

    final_subset_cols = list(best_feat_union) + all_outcomes
    final_subset_cols = list(dict.fromkeys(final_subset_cols))

    final_subset_df = data[final_subset_cols].copy()
    subset_csv_path = os.path.join(output_dir, "best_features_overall_subset.csv")
    final_subset_df.to_csv(subset_csv_path, sep=';', index=True)

    print("\n[INFO] Generating unified recall score heatmap across classification outcomes & methods...")
    recall_heatmap_path = os.path.join(output_dir, "recall_scores_heatmap.pdf")
    plt.figure(figsize=(1.5*len(method_names_cls), 1.2*len(classification_outcomes)))
    class_recall_df = recall_df.loc[classification_outcomes, method_names_cls]
    sns.heatmap(class_recall_df, annot=True, cmap='cividis', fmt=".2f")
    plt.title("recall Scores Heatmap (Classification Outcomes vs. Methods)")
    plt.xlabel("Methods")
    plt.ylabel("Outcomes")
    plt.tight_layout()
    plt.savefig(recall_heatmap_path)
    plt.close()

    print("\n[INFO] Done! Confusion matrix colored white->darkblue (Blues), radial plots colored darkviolet.\n")

if __name__ == "__main__":
    main()

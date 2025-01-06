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
    accuracy_score,
    recall_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

# ADDED: MLPClassifier
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

    # 1) Load Data, create correlation matrix once
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

    # 2) We consider 'gestational_age_delivery' AND 'newborn_weight' as continuous
    #    We'll do classification for the rest, including 'preeclampsia_onset'.
    all_outcomes = [
        "gestational_age_delivery",  # continuous
        "newborn_weight",            # continuous
        "preeclampsia_onset",        # classification
        "delivery_type",
        "newborn_vital_status",
        "newborn_malformations",
        "eclampsia_hellp",
        "iugr"
    ]
    continuous_outcomes = ["gestational_age_delivery", "newborn_weight"]
    classification_outcomes = [o for o in all_outcomes if o not in continuous_outcomes]

    outcome_colors = {
        'gestational_age_delivery': 'red',
        'newborn_weight': 'green',
        'preeclampsia_onset': 'navy',
        'delivery_type': 'blue',
        'newborn_vital_status': 'purple',
        'newborn_malformations': 'orange',
        'eclampsia_hellp': 'cyan',
        'iugr': 'brown'
    }
    fallback_color = "gray"

    # 3) Classification methods
    classifiers = {
        "LogisticRegression": LogisticRegression(
            penalty=None, dual=False, random_state=15,
            solver='newton-cholesky', max_iter=100
        ),
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "GaussianNB": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, algorithm='auto'),
        "Decision Tree": DecisionTreeClassifier(random_state=15),
        "Random Forest": RandomForestClassifier(bootstrap=False, random_state=15),
        "GradientBoosting": GradientBoostingClassifier(max_depth=5, random_state=15),
        "SVM": SVC(probability=True, random_state=15),
        # ADDED MLP
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=15)
    }
    method_names = list(classifiers.keys())

    # Regressors for continuous
    regressors = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # We'll store recall for classification
    recall_df = pd.DataFrame(0.0, index=all_outcomes, columns=method_names)
    method_importances = {out: {} for out in all_outcomes}
    method_conf_matrices = {out: {} for out in classification_outcomes}

    # Classification loop
    for outcome_col in classification_outcomes:
        print(f"\n=== Classification for outcome: {outcome_col} ===")
        X = data.drop(columns=[outcome_col])
        others = [o for o in all_outcomes if o != outcome_col]
        for o_ in others:
            if o_ in X.columns:
                X = X.drop(columns=[o_])
        y = data[outcome_col].copy()

        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=15, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=15
            )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        feat_names = np.array(X.columns)

        for model_name, clf in classifiers.items():
            print(f"  Training {model_name} for {outcome_col} ...")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Use recall
            rec_ = recall_score(y_test, y_pred, average='macro')
            recall_df.loc[outcome_col, model_name] = rec_

            conf_ = confusion_matrix(y_test, y_pred)
            method_conf_matrices[outcome_col][model_name] = conf_

            try:
                perm_res = permutation_importance(
                    clf, X_test, y_test, n_repeats=5, random_state=15, n_jobs=-1
                )
                importances = perm_res.importances_mean
            except Exception as e:
                print(f"[WARN] Permutation importance failed for {model_name}/{outcome_col}: {e}")
                importances = np.zeros(len(feat_names))

            method_importances[outcome_col][model_name] = importances

    # Regression loop
    for outcome_col in continuous_outcomes:
        print(f"\n=== Regression for outcome: {outcome_col} ===")
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
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        feat_names = np.array(X.columns)

        for regr_name, regr in regressors.items():
            print(f"  Training {regr_name} for {outcome_col} ...")
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)
            mse_ = mean_squared_error(y_test, y_pred)
            r2_ = r2_score(y_test, y_pred)
            print(f"    MSE={mse_:.2f}, R2={r2_:.2f}")

            try:
                perm_res = permutation_importance(
                    regr, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
                )
                importances = perm_res.importances_mean
            except Exception as e:
                print(f"[WARN] Perm. importance failed for {regr_name}/{outcome_col}: {e}")
                importances = np.zeros(len(feat_names))

            method_importances[outcome_col][regr_name] = importances

    # 3) Generate combined confusion, permutation, radial
    classification_only = [o for o in classification_outcomes]
    for outcome_col in classification_only:
        # pdfA
        pdfA_path = os.path.join(output_dir, f"pdfA_{outcome_col}.pdf")
        n_methods = len(method_names)
        if n_methods <= 4:
            nrows, ncols = 1, n_methods
        elif n_methods <= 6:
            nrows, ncols = 2, 3
        else:
            nrows, ncols = 2, 4

        figA, axesA = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
        if nrows*ncols == 1:
            axesA = [axesA]
        else:
            axesA = axesA.flatten()

        for i, method_name in enumerate(method_names):
            confm = method_conf_matrices[outcome_col][method_name]
            ax_ = axesA[i]
            sns.heatmap(confm, annot=True, cmap='Blues', fmt='g', ax=ax_)
            ax_.set_title(method_name, fontsize=10)
            ax_.set_xlabel("Predicted")
            ax_.set_ylabel("Actual")

        for j in range(i+1, nrows*ncols):
            axesA[j].axis("off")

        figA.suptitle(f"Combined Confusion Matrices - {outcome_col}", fontsize=14)
        plt.tight_layout()
        figA.savefig(pdfA_path)
        plt.close(figA)

        # pdfB => bar charts, but now we increase the figure's HEIGHT
        pdfB_path = os.path.join(output_dir, f"pdfB_{outcome_col}.pdf")
        nrowsB, ncolsB = nrows, ncols
        figB, axesB = plt.subplots(nrows=nrowsB, ncols=ncolsB, figsize=(6*ncolsB, 6*nrowsB))  # <== increased height
        if nrowsB*ncolsB == 1:
            axesB = [axesB]
        else:
            axesB = axesB.flatten()

        Xtemp = data.drop(columns=[outcome_col])
        for o_ in all_outcomes:
            if o_ != outcome_col and o_ in Xtemp.columns:
                Xtemp = Xtemp.drop(columns=[o_])
        feat_names = list(Xtemp.columns)

        for i, method_name in enumerate(method_names):
            imps = method_importances[outcome_col][method_name]
            sorted_idx = np.argsort(imps)[::-1]
            ax_ = axesB[i]
            feat_labels_temp = [feat_names[k] for k in sorted_idx]
            ax_.barh(feat_labels_temp, imps[sorted_idx], color='skyblue')
            ax_.invert_yaxis()
            ax_.set_title(method_name, fontsize=10)
            ax_.set_xlabel("Mean Decrease")

        for j in range(i+1, nrowsB*ncolsB):
            axesB[j].axis("off")

        figB.suptitle(f"Permutation Importances - {outcome_col}", fontsize=14)
        plt.tight_layout()
        figB.savefig(pdfB_path)
        plt.close(figB)

        # pdfC => radial
        pdfC_path = os.path.join(output_dir, f"pdfC_{outcome_col}.pdf")
        if n_methods <= 4:
            nrowsC, ncolsC = 1, n_methods
        elif n_methods <= 6:
            nrowsC, ncolsC = 2, 3
        else:
            nrowsC, ncolsC = 2, 4

        figC, axesC = plt.subplots(nrows=nrowsC, ncols=ncolsC,
                                   figsize=(6*ncolsC, 5*nrowsC),
                                   subplot_kw=dict(polar=True))
        if nrowsC*ncolsC == 1:
            axesC = [axesC]
        else:
            axesC = axesC.flatten()

        all_imps_ = [method_importances[outcome_col][m] for m in method_names]
        global_max = np.max([np.max(a) for a in all_imps_]) if all_imps_ else 1.0
        base_angles = np.linspace(0, 2*math.pi, len(all_imps_[0]), endpoint=False).tolist()

        for i, method_name in enumerate(method_names):
            imps = method_importances[outcome_col][method_name]
            sorted_idx = np.argsort(imps)[::-1]
            sorted_imps = imps[sorted_idx]
            short_labels_for_plot = [f"F{k+1}" for k in sorted_idx]

            rep_imps = np.concatenate((sorted_imps, [sorted_imps[0]]))
            sub_angles = base_angles[:]
            sub_angles += sub_angles[:1]

            ax_ = axesC[i]
            the_color = outcome_colors.get(outcome_col, fallback_color)
            ax_.plot(sub_angles, rep_imps, linewidth=2, linestyle='solid', color=the_color)
            ax_.fill(sub_angles, rep_imps, alpha=0.25, color=the_color)
            ax_.set_theta_offset(math.pi/2)
            ax_.set_theta_direction(-1)
            degs = np.degrees(sub_angles[:-1])
            ax_.set_thetagrids(degs, labels=short_labels_for_plot, fontsize=6)
            ax_.set_ylim(0, max(global_max, 0))
            ax_.set_title(method_name, fontsize=10)

        for j in range(i+1, nrowsC*ncolsC):
            axesC[j].axis("off")

        figC.suptitle(f"Radial Importances (Unified Scale) - {outcome_col}", fontsize=14)
        legend_elems = []
        the_color = outcome_colors.get(outcome_col, fallback_color)
        line = plt.Line2D([0], [0], color=the_color, lw=2, label=outcome_col)
        legend_elems.append(line)
        figC.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout(rect=[0,0.05,1,1])
        figC.savefig(pdfC_path)
        plt.close(figC)

    # If continuous, do radial for reg
    for outcome_col in continuous_outcomes:
        reg_method_names = list(regressors.keys())
        pdfC_reg_path = os.path.join(output_dir, f"pdfC_{outcome_col}_regression.pdf")

        if len(reg_method_names) <= 4:
            nrowsR, ncolsR = 1, len(reg_method_names)
        else:
            nrowsR, ncolsR = 2, 2

        figCreg, axesCreg = plt.subplots(nrows=nrowsR, ncols=ncolsR,
                                         figsize=(6*ncolsR, 5*nrowsR),
                                         subplot_kw=dict(polar=True))
        if nrowsR*ncolsR == 1:
            axesCreg = [axesCreg]
        else:
            axesCreg = axesCreg.flatten()

        all_imps_r = [method_importances[outcome_col][r] for r in reg_method_names]
        global_max_r = np.max([np.max(a) for a in all_imps_r]) if all_imps_r else 1.0
        base_angles_r = np.linspace(0, 2*math.pi, len(all_imps_r[0]), endpoint=False).tolist()

        for i, r_name in enumerate(reg_method_names):
            imps = method_importances[outcome_col][r_name]
            sorted_idx = np.argsort(imps)[::-1]
            sorted_imps = imps[sorted_idx]
            short_labels_for_plot = [f"F{k+1}" for k in sorted_idx]

            rep_imps = np.concatenate((sorted_imps, [sorted_imps[0]]))
            sub_angles = base_angles_r[:]
            sub_angles += sub_angles[:1]

            ax_ = axesCreg[i]
            the_color = outcome_colors.get(outcome_col, fallback_color)
            ax_.plot(sub_angles, rep_imps, linewidth=2, linestyle='solid', color=the_color)
            ax_.fill(sub_angles, rep_imps, alpha=0.25, color=the_color)
            ax_.set_theta_offset(math.pi/2)
            ax_.set_theta_direction(-1)
            degs = np.degrees(sub_angles[:-1])
            ax_.set_thetagrids(degs, labels=short_labels_for_plot, fontsize=6)
            ax_.set_ylim(0, max(global_max_r, 0))
            ax_.set_title(r_name, fontsize=10)

        for j in range(i+1, nrowsR*ncolsR):
            axesCreg[j].axis("off")

        figCreg.suptitle(f"Radial Importances (Regression) - {outcome_col}", fontsize=14)
        legend_elems_r = []
        the_color_r = outcome_colors.get(outcome_col, fallback_color)
        line_r = plt.Line2D([0],[0], color=the_color_r, lw=2, label=outcome_col)
        legend_elems_r.append(line_r)
        figCreg.legend(handles=legend_elems_r, loc="lower center", bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout(rect=[0,0.05,1,1])
        figCreg.savefig(pdfC_reg_path)
        plt.close(figCreg)

    # 4) One final recall heatmap (only classification)
    print("\n[INFO] Generating unified RECALL score heatmap across classification outcomes & methods...")
    recall_heatmap_path = os.path.join(output_dir, "recall_scores_heatmap.pdf")
    plt.figure(figsize=(1.5*len(method_names), 1.2*len(classification_outcomes)))
    class_recall_df = recall_df.loc[classification_outcomes, method_names]
    sns.heatmap(class_recall_df, annot=True, cmap='viridis', fmt=".2f")
    plt.title("Recall Scores Heatmap (Classification Outcomes vs. Methods)")
    plt.xlabel("Methods")
    plt.ylabel("Outcomes")
    plt.tight_layout()
    plt.savefig(recall_heatmap_path)
    plt.close()

    # 5) pdfD => one heatmap per method, "inferno" colormap
    all_methods = list(classifiers.keys()) + list(regressors.keys())

    for method_name in all_methods:
        # gather all outcomes that have method_name
        applicable_outcomes = []
        all_vals = []
        max_features = 0

        for out_ in all_outcomes:
            if method_name in method_importances[out_]:
                arr_ = method_importances[out_][method_name]
                max_features = max(max_features, len(arr_))
                all_vals.extend(arr_)
                applicable_outcomes.append(out_)

        if not applicable_outcomes:
            continue

        mat = np.zeros((len(applicable_outcomes), max_features))
        row_labels = list(applicable_outcomes)

        global_minD = min(all_vals) if all_vals else 0.0
        global_maxD = max(all_vals) if all_vals else 1.0

        for i, out_ in enumerate(applicable_outcomes):
            arr_ = method_importances[out_][method_name]
            mat[i, :len(arr_)] = arr_

        col_labels = [f"feature_{i}" for i in range(max_features)]
        pdfD_path = os.path.join(output_dir, f"pdfD_{method_name}.pdf")
        plt.figure(figsize=(1.2*max_features, 1.2*len(applicable_outcomes)))
        sns.heatmap(mat, annot=False, cmap='inferno',
                    xticklabels=col_labels, yticklabels=row_labels,
                    vmin=global_minD, vmax=global_maxD)
        plt.title(f"Inferno Heatmap - {method_name} (Importances)")
        plt.xlabel("Features")
        plt.ylabel("Outcomes")
        plt.tight_layout()
        plt.savefig(pdfD_path)
        plt.close()

    print(f"\n[INFO] Done!\n"
          f"For each classification outcome, we created:\n"
          f"  1) pdfA_<outcome>.pdf => combined confusion matrices\n"
          f"  2) pdfB_<outcome>.pdf => combined bar chart importances (height increased)\n"
          f"  3) pdfC_<outcome>.pdf => radial plot\n"
          f"For continuous outcomes, separate regression approach.\n"
          f"We added 'MLP' as a new classifier.\n"
          f"Finally, we produce 'pdfD_<method>.pdf' for each method => inferno heatmap with unified scale.\n"
          f"And 'recall_scores_heatmap.pdf' with macro-Recall across classification outcomes vs. methods.\n")

if __name__ == "__main__":
    main()

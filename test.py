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

# MLP
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

###############################################################################
# Minimal "repel" approach for radial
###############################################################################
def repel_labels(ax, angles, labels):
    """
    Attempt to offset overlapping labels (a minimal approach).
    We'll check distances in angle space & offset them slightly if they are too close.
    This is not a perfect approach but a demonstration of a minimal "repel."
    """
    if len(labels) < 2:
        return
    # angles are len(...) = len(labels). We'll store new angles in array:
    new_angles = angles[:]
    offset = 0.03  # small offset in radians
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if abs(new_angles[i] - new_angles[j]) < offset/2:
                # push one label slightly
                new_angles[j] += offset
    # Now we place them manually
    for i, lbl in enumerate(labels):
        # We convert angle to x,y
        ang = new_angles[i]
        x = 1.05 * math.cos(ang)  # radius slightly >1
        y = 1.05 * math.sin(ang)
        ax.text( x, y, lbl, fontsize=6, ha='center', va='center',
                 rotation=math.degrees(ang - math.pi/2))

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

    # 1) Load Data
    data = pd.read_csv(input_csv, delimiter=';', index_col='id')

    # correlation matrix (with 'seismic')
    corr_matrix = data.corr()
    plt.figure(figsize=(30, 24))
    ax = sns.heatmap(corr_matrix, annot=True, cmap='seismic', fmt='.2f',
                     annot_kws={"size":12}, linewidths=0.5, linecolor='white',
                     cbar_kws={'shrink':0.5}, vmin=-1, vmax=1)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Correlation Matrix (Full Data)', fontsize=20)
    corr_path = os.path.join(output_dir, 'full_data_corr_matrix.pdf')
    plt.savefig(corr_path, bbox_inches='tight')
    plt.close()

    # outcomes
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

    # classifiers
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
    method_names = list(classifiers.keys())

    regressors = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # store classification recall, regression metrics
    recall_df = pd.DataFrame(0.0, index=all_outcomes, columns=method_names)
    regression_metrics = {}

    method_importances = {out: {} for out in all_outcomes}
    method_conf_matrices = {out: {} for out in classification_outcomes}

    ############################
    # Classification pipeline
    ############################
    for outcome_col in classification_outcomes:
        X = data.drop(columns=[outcome_col])
        # remove other outcomes
        others = [o for o in all_outcomes if o != outcome_col]
        for o_ in others:
            if o_ in X.columns:
                X = X.drop(columns=[o_])
        y = data[outcome_col].copy()

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

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        feat_names = np.array(X.columns)

        for m_ in classifiers.keys():
            clf = classifiers[m_]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # store recall
            rec_ = recall_score(y_test, y_pred, average='macro')
            recall_df.loc[outcome_col, m_] = rec_

            # confusion
            conf_ = confusion_matrix(y_test, y_pred)
            method_conf_matrices[outcome_col][m_] = conf_

            # permutation
            try:
                perm_res = permutation_importance(
                    clf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
                )
                importances = perm_res.importances_mean
            except Exception as e:
                importances = np.zeros(len(feat_names))

            method_importances[outcome_col][m_] = importances

    ##########################
    # Regression pipeline
    ##########################
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
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        feat_names = np.array(X.columns)

        regression_metrics[outcome_col] = {}

        for regr_ in regressors.keys():
            model_ = regressors[regr_]
            model_.fit(X_train, y_train)
            y_pred = model_.predict(X_test)

            mse_val = mean_squared_error(y_test, y_pred)
            rmse_val = math.sqrt(mse_val)
            mae_val = mean_absolute_error(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)
            regression_metrics[outcome_col][regr_] = (mse_val, rmse_val, mae_val, r2_val)

            try:
                perm_res = permutation_importance(
                    model_, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
                )
                importances = perm_res.importances_mean
            except:
                importances = np.zeros(len(feat_names))

            method_importances[outcome_col][regr_] = importances

    ##########################
    # Confusion matrix plots (with default colors)
    ##########################
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
        n_methods = len(classifiers)
        nrows, ncols = decide_layout(n_methods)
        figA, axesA = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
        if nrows*ncols == 1:
            axesA = [axesA]
        else:
            axesA = axesA.flatten()

        i_ = 0
        for m_ in classifiers.keys():
            conf_ = method_conf_matrices[outcome_col][m_]
            ax_ = axesA[i_]
            i_ += 1
            # default color
            sns.heatmap(conf_, annot=True, fmt='g', ax=ax_)
            ax_.set_title(m_, fontsize=10)
            ax_.set_xlabel("Predicted")
            ax_.set_ylabel("Actual")

        for j in range(i_, nrows*ncols):
            axesA[j].axis("off")

        plt.tight_layout()
        figA.suptitle(f"Combined Confusion Matrices - {outcome_col}")
        plt.savefig(pdfA_path)
        plt.close(figA)

    ##########################
    # Bar plots, radial, etc. for classification
    ##########################
    for outcome_col in classification_outcomes:
        pdfB_path = os.path.join(output_dir, f"pdfB_{outcome_col}.pdf")
        n_methods = len(classifiers)
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
        for m_ in classifiers.keys():
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

        # compute base angles once
        # #features depends on the shape of Xtemp
        for o_ in all_outcomes:
            if o_ != outcome_col and o_ in Xtemp.columns:
                Xtemp = Xtemp.drop(columns=[o_])
        feat_names_for_radial = np.array(Xtemp.columns)
        n_feat = len(feat_names_for_radial)
        base_angles = np.linspace(0, 2*math.pi, n_feat, endpoint=False).tolist()

        i_ = 0
        for m_ in classifiers.keys():
            imps = method_importances[outcome_col][m_]
            sorted_idx = np.argsort(imps)[::-1]
            sorted_imps = imps[sorted_idx]
            sorted_feats = feat_names_for_radial[sorted_idx]

            rep_imps = np.concatenate((sorted_imps, [sorted_imps[0]]))
            # angles
            angles = base_angles[:]
            angles += angles[:1]

            ax_ = axesC[i_]
            i_ += 1
            the_color = 'gray'
            ax_.plot(angles, rep_imps, linewidth=2, linestyle='solid', color=the_color)
            ax_.fill(angles, rep_imps, alpha=0.25, color=the_color)
            ax_.set_theta_offset(math.pi/2)
            ax_.set_theta_direction(-1)
            ax_.set_ylim(0, max(0, max(rep_imps)))
            ax_.set_title(m_, fontsize=10)

            # minimal "repel": we'll pass angles (except last), label them
            # We do angles[:-1] for the label angles
            label_angles = angles[:-1]
            # We'll call repel_labels
            repel_labels(ax_, label_angles, sorted_feats)

        for j in range(i_, nrowsC*ncolsC):
            axesC[j].axis("off")

        figC.suptitle(f"Radial Importances (Minimal Repel) - {outcome_col}")
        plt.tight_layout()
        plt.savefig(pdfC_path)
        plt.close(figC)

    ##########################
    # Do the same for regression radial
    ##########################
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

        # get the feature set
        Xtemp = data.drop(columns=[outcome_col])
        for o_ in all_outcomes:
            if o_ != outcome_col and o_ in Xtemp.columns:
                Xtemp = Xtemp.drop(columns=[o_])
        feat_names_for_radial = np.array(Xtemp.columns)
        n_feat = len(feat_names_for_radial)
        base_angles_r = np.linspace(0, 2*math.pi, n_feat, endpoint=False).tolist()

        i_ = 0
        for regr_ in reg_method_names:
            imps = method_importances[outcome_col][regr_]
            sorted_idx = np.argsort(imps)[::-1]
            sorted_imps = imps[sorted_idx]
            sorted_feats = feat_names_for_radial[sorted_idx]

            rep_imps = np.concatenate((sorted_imps, [sorted_imps[0]]))
            sub_angles = base_angles_r[:]
            sub_angles += sub_angles[:1]

            ax_ = axesCreg[i_]
            i_ += 1
            the_color = 'gray'
            ax_.plot(sub_angles, rep_imps, linewidth=2, linestyle='solid', color=the_color)
            ax_.fill(sub_angles, rep_imps, alpha=0.25, color=the_color)
            ax_.set_theta_offset(math.pi/2)
            ax_.set_theta_direction(-1)
            ax_.set_ylim(0, max(0, max(rep_imps)))
            ax_.set_title(regr_, fontsize=10)

            # minimal repel
            repel_labels(ax_, sub_angles[:-1], sorted_feats)

        for j in range(i_, nrowsR*ncolsR):
            axesCreg[j].axis("off")

        figCreg.suptitle(f"Radial Importances (Minimal Repel) - {outcome_col}")
        plt.tight_layout()
        plt.savefig(pdfC_reg_path)
        plt.close(figCreg)

    ##########################
    #  Heatmap D: unify scale across all methods
    ##########################
    # 1) find global min, max across all method importances (all outcomes)
    globalD_min = 9999999.0
    globalD_max = -9999999.0

    all_methods = list(classifiers.keys()) + list(regressors.keys())
    for method_name in all_methods:
        for out_ in all_outcomes:
            if method_name in method_importances[out_]:
                arr_ = method_importances[out_][method_name]
                if len(arr_):
                    arr_min = min(arr_)
                    arr_max = max(arr_)
                    if arr_min < globalD_min:
                        globalD_min = arr_min
                    if arr_max > globalD_max:
                        globalD_max = arr_max

    # 2) produce pdfD with features as columns, outcomes as rows
    # rotate feature names 90° at top
    for method_name in all_methods:
        # gather
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

        for i, out_ in enumerate(applicable_outcomes):
            arr_ = method_importances[out_][method_name]
            mat[i, :len(arr_)] = arr_

        col_labels = [f"feat_{i}" for i in range(max_features)]
        pdfD_path = os.path.join(output_dir, f"pdfD_{method_name}.pdf")
        plt.figure(figsize=(1.2*max_features, 1.2*len(applicable_outcomes)))
        # unify scale = globalD_min, globalD_max
        sns.heatmap(mat, annot=False, cmap='inferno',
                    xticklabels=col_labels, yticklabels=row_labels,
                    vmin=globalD_min, vmax=globalD_max)
        plt.xticks(rotation=90)
        plt.title(f"Inferno Heatmap - {method_name} (Importances), Unif Scale")
        plt.xlabel("Features")
        plt.ylabel("Outcomes")
        plt.tight_layout()
        plt.savefig(pdfD_path)
        plt.close()

    ##########################
    #  Subset approach: "top best features" 
    ##########################
    # We'll do EXACTLY your approach, but at the end we re-run "the entire pipeline" with those subsets
    # We'll store in 'subset' folder

    # We'll define a simple approach: topN=5, or cutoff=0.02
    top_importance_cutoff = 0.02
    subset_dir = os.path.join(output_dir, "subset")
    os.makedirs(subset_dir, exist_ok=True)

    best_features_dict = {}

    def pick_subset_features_importances(outcome_col, method_name):
        # fetch importances
        imps = method_importances[outcome_col][method_name]
        # define Xtemp to get feat names
        Xtemp = data.drop(columns=[outcome_col])
        for o_ in all_outcomes:
            if o_ != outcome_col and o_ in Xtemp.columns:
                Xtemp = Xtemp.drop(columns=[o_])
        feat_names = np.array(Xtemp.columns)
        mask = (imps > top_importance_cutoff)
        if not np.any(mask):
            # fallback to single most important
            top_idx = [np.argmax(imps)]
        else:
            top_idx = np.where(mask)[0]
        return feat_names[top_idx].tolist()

    # classification
    for outcome_col in classification_outcomes:
        best_method = None
        best_val = -999
        for m_ in classifiers.keys():
            rec_ = recall_df.loc[outcome_col, m_]
            if rec_ > best_val:
                best_val = rec_
                best_method = m_
        if best_method:
            feats_ = pick_subset_features_importances(outcome_col, best_method)
            best_features_dict[outcome_col] = feats_

    # regression
    def best_mse_method_for_outcome(out_):
        # find minimal MSE in regression_metrics[out_]
        min_mse = 9999999
        best_m = None
        if out_ in regression_metrics:
            for regr_ in regressors.keys():
                if regr_ in regression_metrics[out_]:
                    (mse_val, rmse_val, mae_val, r2_val) = regression_metrics[out_][regr_]
                    if mse_val < min_mse:
                        min_mse = mse_val
                        best_m = regr_
        return best_m

    for outcome_col in continuous_outcomes:
        best_m = best_mse_method_for_outcome(outcome_col)
        if best_m:
            feats_ = pick_subset_features_importances(outcome_col, best_m)
            best_features_dict[outcome_col] = feats_

    # We'll define a helper to re-run "entire pipeline" but only for these outcomes
    def re_run_pipeline_subset(data, outcome_col, feats, outdir):
        """
        We'll store some minimal info to screen and log, e.g. classification confusion or regression metrics.
        """
        # build smaller data
        subset_cols = feats + [outcome_col]
        if len(subset_cols) != len(set(subset_cols)):
            # ensure no duplicates
            subset_cols = list(set(subset_cols))
        sub_data = data[subset_cols].copy()

        # check classification or continuous
        is_cont = (outcome_col in continuous_outcomes)

        print(f"[SUBSET] Re-running pipeline for {outcome_col}, feats={feats}")
        with open(os.path.join(outdir, f"subset_{outcome_col}_log.txt"), "w") as f_:
            if not is_cont:
                # classification
                X = sub_data.drop(columns=[outcome_col])
                y = sub_data[outcome_col].copy()
                for c_ in X.columns:
                    if X[c_].dtype == 'object':
                        X[c_] = LabelEncoder().fit_transform(X[c_].astype(str))
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, stratify=y, random_state=999
                    )
                except:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
                sc_ = StandardScaler()
                X_train = sc_.fit_transform(X_train)
                X_test = sc_.transform(X_test)

                # evaluate each classifier
                f_.write(f"[SUBSET] Classification for {outcome_col}, feats={feats}\n\n")
                for m_ in classifiers.keys():
                    clf_ = classifiers[m_]
                    clf_.fit(X_train, y_train)
                    y_pred = clf_.predict(X_test)
                    rec_ = recall_score(y_test, y_pred, average='macro')
                    f_.write(f"{m_}: recall={rec_:.2f}\n")

            else:
                # regression
                X = sub_data.drop(columns=[outcome_col])
                y = sub_data[outcome_col].copy()
                for c_ in X.columns:
                    if X[c_].dtype == 'object':
                        X[c_] = LabelEncoder().fit_transform(X[c_].astype(str))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
                sc_ = StandardScaler()
                X_train = sc_.fit_transform(X_train)
                X_test = sc_.transform(X_test)

                f_.write(f"[SUBSET] Regression for {outcome_col}, feats={feats}\n\n")
                for regr_ in regressors.keys():
                    model_ = regressors[regr_]
                    model_.fit(X_train, y_train)
                    y_pred = model_.predict(X_test)
                    mse_val = mean_squared_error(y_test, y_pred)
                    rmse_val = math.sqrt(mse_val)
                    mae_val = mean_absolute_error(y_test, y_pred)
                    r2_val = r2_score(y_test, y_pred)
                    f_.write(f"{regr_}: MSE={mse_val:.2f}, RMSE={rmse_val:.2f}, MAE={mae_val:.2f}, R2={r2_val:.2f}\n")

    # finally we run the subset pipeline for each outcome
    subset_out = os.path.join(output_dir, "subset")
    os.makedirs(subset_out, exist_ok=True)
    for out_ in all_outcomes:
        if out_ in best_features_dict:
            sub_feats = best_features_dict[out_]
            re_run_pipeline_subset(data, out_, sub_feats, subset_out)

    print("\n[INFO] Done with the entire pipeline, using default confusion-colors, minimal label repel in radial,")
    print("[INFO] unified scale in pdfD, and repeated subset pipeline in 'subset/' folder. Enjoy!")


if __name__ == "__main__":
    main()

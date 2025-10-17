#!/usr/bin/env python
# coding: utf‑8
###############################################################################
# 0. GLOBAL SEED  -------------------------------------------------------------
###############################################################################
SEED = 7
import sys, os, subprocess, argparse, math, random, warnings
import numpy as np
np.random.seed(SEED); random.seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)
ENV_NAME = "ml_preeclampsia"

###############################################################################
# 1. CONDA ENV HELPERS  -------------------------------------------------------
###############################################################################
def conda_env_exists(env_name=ENV_NAME):
    try:
        out = subprocess.run(["conda", "env", "list"],
                             capture_output=True, text=True, check=True).stdout
        return any(line.startswith(env_name + " ") or f"/{env_name}" in line
                   for line in out.splitlines())
    except subprocess.CalledProcessError as e:
        print(f"[ERR] checking conda envs: {e}"); sys.exit(1)

def create_conda_env_if_needed(env_name=ENV_NAME):
    if conda_env_exists(env_name):
        print(f"[INFO] Conda env '{env_name}' already exists."); return
    print(f"[INFO] Creating env '{env_name}' …")
    subprocess.run(["conda", "create", "-y", "-n", env_name, "python=3.9"],
                   check=True)
    pkgs = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "tqdm"]
    subprocess.run(["conda", "run", "-n", env_name, "pip", "install"] + pkgs,
                   check=True)
    print(f"[INFO] Env '{env_name}' created.")

###############################################################################
# 2. LIBRARIES  ---------------------------------------------------------------
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    "font.size":      17,
    "axes.titlesize": 20,
    "axes.labelsize": 20
})
plt.rcParams['axes.unicode_minus'] = True        # use the real “−” on axes

import shutil
def perm_importance_safe(model, X, y, repeats=5):
    """Permutation importance that works on Windows without WMIC."""
    # If on Windows and `wmic` is absent, do single‑threaded.
    n_jobs = -1
    if os.name == "nt" and shutil.which("wmic") is None:
        n_jobs = 1
    try:
        res = permutation_importance(model, X, y,
                                     n_repeats=repeats,
                                     random_state=SEED,
                                     n_jobs=n_jobs)
        return res.importances_mean
    except Exception:
        # absolute fallback: run serially even if the first attempt failed
        res = permutation_importance(model, X, y,
                                     n_repeats=repeats,
                                     random_state=SEED,
                                     n_jobs=1)
        return res.importances_mean

###############################################################################
# 3. ARGUMENT PARSER  ---------------------------------------------------------
###############################################################################
def parse_arguments():
    p = argparse.ArgumentParser(description="Deterministic ML pipeline.")
    p.add_argument("--install_conda", action="store_true",
                   help="Create conda env then re‑run script inside it.")
    p.add_argument("--input",  type=str, required=False, help="Path to CSV/TSV")
    p.add_argument("--output", type=str, required=False, help="Output folder")
    return p.parse_args()

###############################################################################
# 4. MAIN  --------------------------------------------------------------------
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
        print("[ERROR] --input and --output are mandatory"); sys.exit(1)
    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 4.1 Read data (smart delimiter)
    # ------------------------------------------------------------------ #
    def smart_read(path):
        for d in [';', ',', '\t']:
            df = pd.read_csv(path, delimiter=d)
            if df.shape[1] > 1:
                return df
        raise ValueError("Unable to detect delimiter.")

    df = smart_read(args.input)
    data = df.set_index('id') if 'id' in df.columns else df
    if data.columns[0].startswith('Unnamed'):
        data = data.drop(columns=data.columns[0])
    data.index.name = data.index.name or "index"

    # OUTCOMES & FEATURES ------------------------------------------------
    outcomes = ["gestational_age_delivery", "newborn_weight",
                "preeclampsia_onset", "delivery_type", "newborn_vital_status",
                "newborn_malformations", "eclampsia_hellp", "iugr"]
    continuous_out  = ["gestational_age_delivery", "newborn_weight"]
    classification_out = [o for o in outcomes if o not in continuous_out]
    feature_cols = list(data.drop(columns=outcomes, errors='ignore').columns)

    # ------------------------------------------------------------------ #
    # 5.  CORRELATION MATRICES
    # ------------------------------------------------------------------ #
    def draw_corr(df_corr, title, fname, ann_size, figsize):
        plt.figure(figsize=figsize)
        annot_txt = df_corr.round(2).astype(str).replace('-', '−', regex=True)
        sns.heatmap(df_corr, annot=annot_txt, fmt='',
                    cmap="seismic",
                    vmin=-1, vmax=1, linewidths=.4, linecolor='white',
                    cbar_kws={"shrink":0.6}, annot_kws={"size":ann_size})
        plt.xticks(rotation=90, ha='right', fontsize=ann_size)
        plt.yticks(fontsize=ann_size)
        plt.title(title, pad=30)
        plt.tight_layout()
        plt.savefig(f"{args.output}/{fname}")
        plt.close()

    # full matrix
    draw_corr(data.corr(numeric_only=True),
              "Correlation Matrix", "full_corr_matrix.pdf",
              ann_size=13, figsize=(32,26))

    # filtered matrix for paper (include more vars, 30 pt font)
    CORR_TH_PAPER = 0.12      # include variables with |ρ| ≥ 0.12 to any outcome
    corr_abs = data.corr(numeric_only=True).abs()
    keep_vars = [c for c in feature_cols
                 if corr_abs.loc[c, outcomes].max() >= CORR_TH_PAPER]
    draw_corr(data[keep_vars + outcomes].corr(numeric_only=True),
              "Filtered Correlation Matrix", "Fig1_paper.pdf",
              ann_size=20, figsize=(28,22))

    # ------------------------------------------------------------------ #
    # 6.  MODEL DICTIONARIES
    # ------------------------------------------------------------------ #
    classifiers = {
        "LogReg": LogisticRegression(max_iter=100, solver='newton-cholesky',
                                     random_state=SEED),
        "LDA":    LinearDiscriminantAnalysis(),
        "GNB":    GaussianNB(),
        "KNN":    KNeighborsClassifier(n_neighbors=5),
        "DecTree": DecisionTreeClassifier(random_state=SEED),
        "RF":     RandomForestClassifier(bootstrap=False, random_state=SEED),
        "GradBoost": GradientBoostingClassifier(max_depth=5, random_state=SEED),
        "SVM":    SVC(probability=True, random_state=SEED),
        "MLP":    MLPClassifier(hidden_layer_sizes=(100,),
                                max_iter=300, random_state=SEED)
    }
    regressors = {
        "RFreg": RandomForestRegressor(n_estimators=100, random_state=SEED),
        "GBreg": GradientBoostingRegressor(n_estimators=100, random_state=SEED)
    }
    meth_cls = list(classifiers.keys())
    meth_reg = list(regressors.keys())

    # storage structures
    recall_df = pd.DataFrame(0.0, index=outcomes, columns=meth_cls)
    reg_metrics = {}
    importances = {o: {} for o in outcomes}
    conf_mats   = {o: {} for o in classification_out}

    # ------------------------------------------------------------------ #
    # 7.  TRAINING – CLASSIFICATION
    # ------------------------------------------------------------------ #
    for out in classification_out:
        X = data[feature_cols].copy()
        y = data[out].copy()

        # encode object columns
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2,
            stratify=y if len(y.unique()) > 1 else None,
            random_state=SEED
        )
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        for m in meth_cls:
            model = classifiers[m]
            model.fit(Xtr, ytr)
            ypred = model.predict(Xte)

            recall_df.loc[out, m] = recall_score(yte, ypred,
                                                 average='macro')
            conf_mats[out][m] = confusion_matrix(yte, ypred)

            try:
                p_res = permutation_importance(model, Xte, yte,
                                               n_repeats=5,
                                               random_state=SEED, n_jobs=-1)
                importances[out][m] = p_res.importances_mean
            except Exception:
                importances[out][m] = np.zeros(len(feature_cols))

    # ------------------------------------------------------------------ #
    # 8.  TRAINING – REGRESSION
    # ------------------------------------------------------------------ #
    for out in continuous_out:
        X = data[feature_cols].copy()
        y = data[out].copy()

        # encode
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        reg_metrics[out] = {}
        for m in meth_reg:
            model = regressors[m]
            model.fit(Xtr, ytr)
            ypred = model.predict(Xte)

            mse = mean_squared_error(yte, ypred)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(yte, ypred)
            r2  = r2_score(yte, ypred)
            reg_metrics[out][m] = (mse, rmse, mae, r2)

            try:
                p_res = permutation_importance(model, Xte, yte,
                                               n_repeats=5,
                                               random_state=SEED, n_jobs=-1)
                importances[out][m] = p_res.importances_mean
            except Exception:
                importances[out][m] = np.zeros(len(feature_cols))

    # ------------------------------------------------------------------ #
    # 9.  FIGURE A – CONFUSION MATRICES
    # ------------------------------------------------------------------ #
    def grid_shape(n):
        if n <= 4:   return (1, n)
        if n <= 6:   return (2, 3)
        if n <= 9:   return (3, 3)
        return (3, 4)

    for out in classification_out:
        nr, nc = grid_shape(len(meth_cls))
        fig, axes = plt.subplots(nr, nc, figsize=(4.4*nc, 3.8*nr))
        axes = axes.flatten()
        for i, m in enumerate(meth_cls):
            sns.heatmap(conf_mats[out][m], annot=True, fmt='g',
                        cmap=plt.cm.Blues, ax=axes[i])
            axes[i].set_title(m, fontsize=18)
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
        for ax in axes[len(meth_cls):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{args.output}/pdfA_{out}.pdf")
        plt.close()

    # ------------------------------------------------------------------ #
    # 10. FIGURE B – IMPORTANCE BAR GRIDS
    # ------------------------------------------------------------------ #
    for out in classification_out:
        nr, nc = grid_shape(len(meth_cls))
        fig, axes = plt.subplots(nr, nc, figsize=(5.5*nc, 7*nr))
        axes = axes.flatten()
        for i, m in enumerate(meth_cls):
            imp = importances[out][m]
            order = np.argsort(imp)[::-1]
            axes[i].barh(np.array(feature_cols)[order], imp[order],
                         color='steelblue')
            axes[i].invert_yaxis()
            axes[i].tick_params(axis='y', labelsize=11)
            axes[i].set_title(m, fontsize=18)
            axes[i].set_xlabel("Mean Decrease")
        for ax in axes[len(meth_cls):]:
            ax.axis("off")
        fig.subplots_adjust(left=0.38)
        plt.tight_layout()
        plt.savefig(f"{args.output}/pdfB_{out}.pdf", bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------ #
    # 11. FIGURE 3 – best‑performing classifier per categorical outcome
    # ------------------------------------------------------------------ #
    ordered_panels = classification_out  # keep original outcome ordering

    # pick model with highest macro‑recall for each outcome
    best_method = {
        o: recall_df.loc[o, meth_cls].idxmax() for o in ordered_panels
    }

    nr, nc = grid_shape(len(ordered_panels))
    fig3, axes3 = plt.subplots(nr, nc, figsize=(18, 14))
    axes3 = axes3.flatten()

    for idx, out in enumerate(ordered_panels):
        model_name = best_method[out]
        imp_vec = importances[out][model_name]
        order = np.argsort(imp_vec)[::-1]
        axes3[idx].barh(np.array(feature_cols)[order], imp_vec[order],
                        color='steelblue')
        axes3[idx].invert_yaxis()
        axes3[idx].tick_params(axis='y', labelsize=10)
        pretty_out = out.replace('_', ' ').title()
        axes3[idx].set_title(
            f"({chr(65+idx)}) {pretty_out} – {model_name}", fontsize=18
        )
        axes3[idx].set_xlabel("Mean Decrease")

    for ax in axes3[len(ordered_panels):]:
        ax.axis("off")

    fig3.subplots_adjust(left=0.38)
    plt.tight_layout()
    plt.savefig(f"{args.output}/Fig3_paper.pdf", bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------ #
    # 12. IMPORTANCES DOT‑PLOT
    # ------------------------------------------------------------------ #
    gmin = min(arr.min() for d in importances.values() for arr in d.values())
    gmax = max(arr.max() for d in importances.values() for arr in d.values())
    norm = plt.Normalize(gmin, gmax); cmap = plt.cm.inferno

    all_methods = meth_cls + meth_reg
    nr, nc = grid_shape(len(all_methods))
    fig_imp, axes_imp = plt.subplots(nr, nc, figsize=(10*nc, 9*nr))
    axes_imp = axes_imp.flatten()
    for j, m in enumerate(all_methods):
        mat = np.zeros((len(outcomes), len(feature_cols)))
        for r, out in enumerate(outcomes):
            if m in importances[out]:
                mat[r, :] = importances[out][m]
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r, c]
                axes_imp[j].scatter(c, r,
                                    s=280*abs(val)/(gmax+1e-9)+8,
                                    color=cmap(norm(val)))
        axes_imp[j].set_xticks(range(len(feature_cols)))
        axes_imp[j].set_xticklabels(feature_cols, rotation=90, fontsize=8)
        axes_imp[j].set_yticks(range(len(outcomes)))
        axes_imp[j].set_yticklabels(outcomes, fontsize=9)
        axes_imp[j].set_title(m, fontsize=13)
    for ax in axes_imp[len(all_methods):]:
        ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = fig_imp.add_axes([0.93, 0.25, 0.02, 0.5])
    fig_imp.colorbar(sm, cax=cax, label="Importance")
    fig_imp.subplots_adjust(wspace=0.45, hspace=0.75,
                            bottom=0.28, right=0.9)
    plt.savefig(f"{args.output}/importances.pdf")
    plt.close()

    # ------------------------------------------------------------------ #
    # 13. RECALL HEAT‑MAP  (Fig 2)
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(1.4*len(meth_cls), 1.2*len(classification_out)))
    sns.heatmap(recall_df.loc[classification_out], annot=True,
                fmt=".2f", cmap="inferno")
    plt.title("Recall Scores (macro)", pad=18)
    plt.xlabel("Methods"); plt.ylabel("Outcomes")
    plt.tight_layout()
    plt.savefig(f"{args.output}/Fig2_paper.pdf")
    plt.close()

    # ------------------------------------------------------------------ #
    # 14. REGRESSION METRICS TXT
    # ------------------------------------------------------------------ #
    with open(f"{args.output}/regression_metrics_summary.txt", "w") as f:
        f.write("Regression Metrics (MSE | RMSE | MAE | R2)\n\n")
        for out in continuous_out:
            f.write(f"{out}:\n")
            for m in meth_reg:
                mse, rmse, mae, r2 = reg_metrics[out][m]
                f.write(f"  {m:<10}  MSE={mse:.2f}  RMSE={rmse:.2f}  "
                        f"MAE={mae:.2f}  R2={r2:.2f}\n")
            f.write("\n")

    # ------------------------------------------------------------------ #
    # 15. FEATURE SUBSETS CSV EXPORT
    # ------------------------------------------------------------------ #
    subset_union = {"02": set(), "50": set(), "25": set()}

    def best_cls(out): return recall_df.loc[out].idxmax()
    def best_reg(out): return min(reg_metrics[out],
                                  key=lambda m: reg_metrics[out][m][0])

    for out in outcomes:
        m = best_cls(out) if out in classification_out else best_reg(out)
        imp = importances[out][m]
        order = np.argsort(imp)[::-1]
        mask = imp > 0.02
        subset_union["02"].update(
            np.array(feature_cols)[mask] if np.any(mask)
            else [feature_cols[order[0]]]
        )
        subset_union["50"].update(
            np.array(feature_cols)[order[:max(1, int(len(order)*0.5))]])
        subset_union["25"].update(
            np.array(feature_cols)[order[:max(1, int(len(order)*0.25))]])

    for tag, feats in subset_union.items():
        data[list(feats) + outcomes].to_csv(
            f"{args.output}/subset_{tag}.csv", sep=';', index=True)

    print(f"\n[INFO] All outputs written to: {args.output}\n")

###############################################################################
if __name__ == "__main__":
    main()

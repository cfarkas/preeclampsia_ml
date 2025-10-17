#!/usr/bin/env python
# coding: utf-8
###############################################################################
# Deterministic ML pipeline for Pregnancy Outcomes and Child Obesity datasets
#
# Adds:
#   --input PATH
#   --output DIR
#   --outcomes A,B,C    (comma-separated; shorthand mapping allowed)
#   --subset            (docx-driven cohort filters)
#   --exclude_3T        (drop 3T predictors in pregnancy dataset)
#   --exclude_cols      (extra predictor columns to drop)
#   --exclude_co_labels (drop both Child-Obesity label columns from X)
#   --drop-missing      (DEFAULT) drop rows w/ NaNs in predictors or outcome
#   --replace-with-zeros replace NaNs in predictors with 0 (drops rows with NaN outcome)
#   --pca               make PC1 vs. PC2 plots colored by each outcome
#
# New in this version:
#   --width_fig1  --height_fig1   (Filtered correlation matrix size)
#   --width_fig2  --height_fig2   (Recall heatmap size)
#   --width_fig3  --height_fig3   (Best-model bar grids size)
#   --imp_dot_scale --imp_dot_min (Control dot sizes in importances grid)
#
# Notes aligned with Info_Datasets_ML.docx:
#   Pregnancy: outcomes = Class_PTB (1=no, 2=yes), Class_Macrosomia (1=no, 2=yes);
#              subsets via Class_GDM (1=no, 2=yes); can exclude 3T predictors.
#   Child Obesity: outcomes = Clase_Sexo_1F_2M, Clase_FTO_0WT_1HET_2MUT;
#                  optional sex/genotype subsets; dataset includes missing data.
#                  This script does NOT impute. Choose a missing-data mode.
###############################################################################

SEED = 7
import sys, os, subprocess, argparse, math, random, warnings, re, csv
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
from sklearn.preprocessing import StandardScaler
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
from sklearn.decomposition import PCA

plt.rcParams.update({
    "figure.dpi":     300,
    "font.size":      17,
    "axes.titlesize": 20,
    "axes.labelsize": 20
})
plt.rcParams['axes.unicode_minus'] = True

###############################################################################
# 3. ARGUMENT PARSER  ---------------------------------------------------------
###############################################################################
def parse_arguments():
    p = argparse.ArgumentParser(description="Deterministic ML pipeline.")
    p.add_argument("--install_conda", action="store_true",
                   help="Create conda env then re-run script inside it.")
    p.add_argument("--input",  type=str, required=False, help="Path to CSV/TSV")
    p.add_argument("--output", type=str, required=False, help="Output folder")
    p.add_argument("--outcomes", type=str, required=False,
                   help=("Comma-separated outcome names (shorthand OK). "
                         "Examples: 'Class_PTB,Class_Macrosomia' or "
                         "'Clase_Sexo_1F_2M,Clase_FTO_0WT_1HET_2MUT'."))

    # dataset-aware helpers from the docx
    p.add_argument("--subset", type=str, default="all",
                   choices=["all", "gdm_yes", "gdm_no", "girls", "boys",
                            "fto_wt", "fto_het", "fto_mut"],
                   help="Filter group (see Info_Datasets_ML.docx).")
    p.add_argument("--exclude_3T", action="store_true",
                   help="Exclude predictors containing '3T' (pregnancy dataset).")
    p.add_argument("--exclude_cols", type=str, default=None,
                   help="Comma-separated list of columns to drop from predictors.")
    p.add_argument("--exclude_co_labels", action="store_true",
                   help=("Child Obesity: drop BOTH label columns from predictors "
                         "so neither Sex nor FTO are used as features."))
    p.add_argument("--id_col", type=str, default=None,
                   help="Optional ID column name (default: auto-detect 'ID'/'id').")

    # No-imputation missing-data strategy
    g = p.add_mutually_exclusive_group()
    g.add_argument("--drop-missing", action="store_true",
                   help="Drop rows with missing values in predictors OR the active outcome (DEFAULT).")
    g.add_argument("--replace-with-zeros", action="store_true",
                   help="Replace NaNs in predictors with 0; still drops rows where the outcome is missing.")

    # PCA visualization
    p.add_argument("--pca", action="store_true",
                   help="Produce PC1 vs PC2 plots, colored by each selected outcome.")

    # --- NEW: figure sizes for Fig1, Fig2, Fig3 ---
    p.add_argument("--width_fig1",  type=float, default=None, help="Width (inches) for Fig1.")
    p.add_argument("--height_fig1", type=float, default=None, help="Height (inches) for Fig1.")
    p.add_argument("--width_fig2",  type=float, default=None, help="Width (inches) for Fig2.")
    p.add_argument("--height_fig2", type=float, default=None, help="Height (inches) for Fig2.")
    p.add_argument("--width_fig3",  type=float, default=None, help="Width (inches) for Fig3.")
    p.add_argument("--height_fig3", type=float, default=None, help="Height (inches) for Fig3.")

    # --- NEW: importances dot-plot scaling ---
    p.add_argument("--imp_dot_scale", type=float, default=900.0,
                   help="Scale factor for dot sizes in importances grid (larger -> bigger dots).")
    p.add_argument("--imp_dot_min",   type=float, default=12.0,
                   help="Minimum dot size in importances grid.")

    return p.parse_args()

###############################################################################
# 4. STRING & OUTCOME HELPERS  ------------------------------------------------
###############################################################################
def smart_read(path):
    """Try ; , or tab as delimiters; fall back to csv.Sniffer."""
    for d in [';', ',', '\t']:
        try:
            df = pd.read_csv(path, delimiter=d)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(4096)
    dialect = csv.Sniffer().sniff(sample)
    return pd.read_csv(path, delimiter=dialect.delimiter)

def normalize_name(s: str) -> str:
    """Lowercase, remove parentheses, non-alnum -> underscores."""
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'[^A-Za-z0-9]+', '_', s)
    return s.strip('_').lower()

def map_outcomes_from_user(user_str: str, columns: list) -> list:
    """Map user-provided tokens to actual column names (shorthand OK)."""
    if not user_str:
        return []
    try:
        tokens = next(csv.reader([user_str]))
    except Exception:
        tokens = [t for t in re.split(r'[;,]', user_str) if t.strip()]
    tokens = [t.strip() for t in tokens if t and t.strip()]
    if not tokens:
        return []
    col_norm_map = {normalize_name(c): c for c in columns}
    resolved = []
    for tok in tokens:
        if tok in columns:
            resolved.append(tok); continue
        subs = [c for c in columns if tok in c]
        if len(subs) == 1:
            resolved.append(subs[0]); continue
        ntok = normalize_name(tok)
        if ntok in col_norm_map:
            resolved.append(col_norm_map[ntok]); continue
        cands = [c for n,c in col_norm_map.items() if n.startswith(ntok)]
        if len(cands) == 1:
            resolved.append(cands[0]); continue
        cands2 = [c for c in columns if c in user_str]
        for c in cands2:
            if c not in resolved:
                resolved.append(c)
    # dedupe preserving order
    seen = set(); final = []
    for c in resolved:
        if c not in seen and c in columns:
            final.append(c); seen.add(c)
    return final

def infer_default_outcomes(cols: list) -> list:
    """Defaults based on docx dataset signatures."""
    po = ["Class_PTB (1=no, 2=yes)", "Class_Macrosomia (1=no, 2=yes)"]
    if any(c in cols for c in po):
        return [c for c in po if c in cols]
    co = ["Clase_Sexo_1F_2M", "Clase_FTO_0WT_1HET_2MUT"]
    if any(c in cols for c in co):
        return [c for c in co if c in cols]
    return []

def is_classification_target(y: pd.Series) -> bool:
    """Heuristic for classification vs regression."""
    if y.dtype.name in ("object", "category", "bool"):
        return True
    try:
        uniq = pd.unique(y.dropna())
        if len(uniq) <= 10:
            arr = uniq.astype(float)
            return np.all(np.isclose(arr, np.round(arr)))
    except Exception:
        pass
    return False

def to_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to numeric; non-convertible -> NaN."""
    X2 = X.copy()
    for c in X2.columns:
        if X2[c].dtype == 'object':
            X2[c] = pd.to_numeric(X2[c], errors='coerce')
    return X2

def as_1d_axes(axes):
    """Return a flattened 1D numpy array of axes (handles single Axes)."""
    import numpy as _np
    return _np.ravel(_np.atleast_1d(axes))

###############################################################################
# 5. MAIN  --------------------------------------------------------------------
###############################################################################
def main():
    args = parse_arguments()

    # optional environment bootstrap
    if args.install_conda:
        create_conda_env_if_needed()
        forward = []
        if args.input:  forward += ["--input", args.input]
        if args.output: forward += ["--output", args.output]
        if args.outcomes: forward += ["--outcomes", args.outcomes]
        if args.subset and args.subset != "all": forward += ["--subset", args.subset]
        if args.exclude_3T: forward += ["--exclude_3T"]
        if args.exclude_cols: forward += ["--exclude_cols", args.exclude_cols]
        if args.exclude_co_labels: forward += ["--exclude_co_labels"]
        if args.id_col: forward += ["--id_col", args.id_col]
        if args.replace_with_zeros: forward += ["--replace-with-zeros"]
        if args.drop_missing: forward += ["--drop-missing"]
        if args.pca: forward += ["--pca"]
        # propagate figure/dot flags if set
        for k in ["width_fig1","height_fig1","width_fig2","height_fig2","width_fig3","height_fig3",
                  "imp_dot_scale","imp_dot_min"]:
            v = getattr(args, k, None)
            if v is not None:
                forward += [f"--{k}", str(v)]
        subprocess.run(["conda", "run", "-n", ENV_NAME, "python", __file__] + forward, check=True)
        sys.exit(0)

    if not args.input or not args.output:
        print("[ERROR] --input and --output are mandatory"); sys.exit(1)
    os.makedirs(args.output, exist_ok=True)

    # missing-data mode (default: drop)
    missing_mode = 'zeros' if args.replace_with_zeros else 'drop'

    # ------------------------------------------------------------------ #
    # 5.1 Read data / set index
    # ------------------------------------------------------------------ #
    df = smart_read(args.input).copy()
    id_col = args.id_col
    if not id_col:
        id_col = "ID" if "ID" in df.columns else ("id" if "id" in df.columns else None)
    data = df.set_index(id_col) if id_col in df.columns else df
    if data.columns[0].startswith('Unnamed'):
        data = data.drop(columns=data.columns[0])
    data.index.name = data.index.name or "index"

    # ------------------------------------------------------------------ #
    # 5.2 Dataset-aware defaults and filters (per docx)
    # ------------------------------------------------------------------ #
    columns = list(data.columns)
    gdm_col = "Class_GDM (1=no, 2=yes)"
    co_label_cols = ["Clase_Sexo_1F_2M", "Clase_FTO_0WT_1HET_2MUT"]

    # Outcomes
    outcomes = map_outcomes_from_user(args.outcomes, columns) or infer_default_outcomes(columns)
    if not outcomes:
        print("[ERROR] Could not infer outcomes. Use --outcomes A,B or check column names.")
        sys.exit(1)

    # Subset filters per docx guidance. :contentReference[oaicite:1]{index=1}
    if args.subset != "all":
        if gdm_col in columns and args.subset in ("gdm_yes", "gdm_no"):
            val = 2 if args.subset == "gdm_yes" else 1
            data = data.loc[data[gdm_col] == val]
            print(f"[INFO] Applied subset: {args.subset} -> {gdm_col} == {val}")
        elif all(c in columns for c in co_label_cols):
            if args.subset == "girls":
                data = data.loc[data["Clase_Sexo_1F_2M"] == 1]
            elif args.subset == "boys":
                data = data.loc[data["Clase_Sexo_1F_2M"] == 2]
            elif args.subset == "fto_wt":
                data = data.loc[data["Clase_FTO_0WT_1HET_2MUT"] == 0]
            elif args.subset == "fto_het":
                data = data.loc[data["Clase_FTO_0WT_1HET_2MUT"] == 1]
            elif args.subset == "fto_mut":
                data = data.loc[data["Clase_FTO_0WT_1HET_2MUT"] == 2]
            print(f"[INFO] Applied subset: {args.subset}")
        else:
            print("[WARN] --subset ignored (dataset columns not found).")

    # Exclusions for predictors
    exclude_cols = []
    if args.exclude_cols:
        try:
            exclude_cols = next(csv.reader([args.exclude_cols]))
        except Exception:
            exclude_cols = [c.strip() for c in re.split(r'[;,]', args.exclude_cols)]
        exclude_cols = [c for c in exclude_cols if c]

    if args.exclude_3T:
        drop_3t = [c for c in columns if "3T" in c]
        exclude_cols.extend(drop_3t)
        print(f"[INFO] Dropping {len(drop_3t)} third-trimester columns (contains '3T').")

    if args.exclude_co_labels and all(c in columns for c in co_label_cols):
        exclude_cols.extend([c for c in co_label_cols if c not in outcomes])
        print(f"[INFO] Excluding Child Obesity label columns from predictors: {co_label_cols}")

    # Final feature columns
    feature_cols = [c for c in data.columns if c not in outcomes and c not in set(exclude_cols)]

    # Sanity checks
    if len(data) < 10:
        print("[ERROR] Not enough rows after filtering (< 10)."); sys.exit(1)
    if len(feature_cols) == 0:
        print("[ERROR] No predictors left after exclusions."); sys.exit(1)

    # Ensure numeric predictors (object -> numeric coercion)
    data[feature_cols] = to_numeric_df(data[feature_cols])

    # ------------------------------------------------------------------ #
    # 5.3  CORRELATION MATRICES
    # ------------------------------------------------------------------ #
    def draw_corr(df_corr, title, fname, ann_size, figsize):
        if df_corr.shape[0] == 0 or df_corr.shape[1] == 0:
            return
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
        plt.savefig(os.path.join(args.output, fname))
        plt.close()

    # Full numeric matrix
    draw_corr(data.corr(numeric_only=True),
              "Correlation Matrix", "full_corr_matrix.pdf",
              ann_size=13, figsize=(max(12, 0.5*len(data.columns)+8), 24))

    # Filtered matrix (|rho| >= 0.12 vs outcomes) -> Fig1
    CORR_TH_PAPER = 0.12
    corr_abs = data.corr(numeric_only=True).abs()
    keep_vars = [c for c in feature_cols
                 if c in corr_abs.index and any(o in corr_abs.columns for o in outcomes)
                 and corr_abs.loc[c, outcomes].max() >= CORR_TH_PAPER]
    safe_block = [c for c in keep_vars if c in data.columns] + [o for o in outcomes if o in data.columns]
    safe_block = list(dict.fromkeys(safe_block))
    if len(safe_block) >= 2:
        w1 = args.width_fig1  if args.width_fig1  is not None else max(12, 0.5*len(safe_block)+8)
        h1 = args.height_fig1 if args.height_fig1 is not None else 22
        draw_corr(data[safe_block].corr(numeric_only=True),
                  "Filtered Correlation Matrix", "Fig1_paper.pdf",
                  ann_size=18, figsize=(w1, h1))

    # ------------------------------------------------------------------ #
    # 5.4  MODEL DICTIONARIES
    # ------------------------------------------------------------------ #
    classifiers = {
        "LogReg":   LogisticRegression(max_iter=500, solver='lbfgs', random_state=SEED),
        "LDA":      LinearDiscriminantAnalysis(),
        "GNB":      GaussianNB(),
        "KNN":      KNeighborsClassifier(n_neighbors=5),
        "DecTree":  DecisionTreeClassifier(random_state=SEED),
        "RF":       RandomForestClassifier(bootstrap=False, random_state=SEED),
        "GradBoost":GradientBoostingClassifier(max_depth=5, random_state=SEED),
        "SVM":      SVC(probability=True, random_state=SEED),
        "MLP":      MLPClassifier(hidden_layer_sizes=(100,), max_iter=400, random_state=SEED)
    }
    regressors = {
        "RFreg": RandomForestRegressor(n_estimators=200, random_state=SEED),
        "GBreg": GradientBoostingRegressor(n_estimators=200, random_state=SEED)
    }
    meth_cls = list(classifiers.keys())
    meth_reg = list(regressors.keys())

    # Outcome types
    classification_out = [o for o in outcomes if is_classification_target(data[o])]
    continuous_out     = [o for o in outcomes if o not in classification_out]

    # storage
    recall_df   = pd.DataFrame(0.0, index=classification_out, columns=meth_cls)
    reg_metrics = {}
    importances = {o: {} for o in outcomes}
    conf_mats   = {o: {} for o in classification_out}

    # helper: missing-data strategy per outcome
    def prepare_xy_for_outcome(out_col: str):
        X = data[feature_cols].copy()
        y = data[out_col].copy()
        if missing_mode == 'drop':
            XY = pd.concat([X, y], axis=1).dropna()
            Xc = XY[feature_cols]
            yc = XY[out_col]
            dropped = len(X) - len(Xc)
            if dropped > 0:
                print(f"[INFO] Missing-mode=drop: removed {dropped} rows for '{out_col}'.")
        else:
            Xc = X.fillna(0)
            mask = y.notna()
            Xc = Xc.loc[mask]
            yc = y.loc[mask]
            dropped = len(X) - len(Xc)
            if dropped > 0:
                print(f"[INFO] Missing-mode=zeros: removed {dropped} rows due to missing '{out_col}'.")
        return Xc, yc

    # ------------------------------------------------------------------ #
    # 5.5  TRAINING – CLASSIFICATION
    # ------------------------------------------------------------------ #
    for out in classification_out:
        Xc, yc = prepare_xy_for_outcome(out)
        if len(Xc) < 10 or len(pd.unique(yc)) < 2:
            print(f"[WARN] Skipping classification for '{out}' (insufficient samples/classes).")
            continue

        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            Xc, yc, test_size=0.2,
            stratify=yc if len(pd.unique(yc)) > 1 else None,
            random_state=SEED
        )
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_raw)
        Xte = scaler.transform(Xte_raw)

        for m in meth_cls:
            model = classifiers[m]
            model.fit(Xtr, ytr)
            ypred = model.predict(Xte)

            recall_df.loc[out, m] = recall_score(yte, ypred, average='macro')
            conf_mats[out][m] = confusion_matrix(yte, ypred)

            try:
                p_res = permutation_importance(model, Xte, yte,
                                               n_repeats=5,
                                               random_state=SEED, n_jobs=-1)
                importances[out][m] = p_res.importances_mean
            except Exception:
                importances[out][m] = np.zeros(len(feature_cols))

    # ------------------------------------------------------------------ #
    # 5.6  TRAINING – REGRESSION
    # ------------------------------------------------------------------ #
    for out in continuous_out:
        Xc, yc = prepare_xy_for_outcome(out)
        if len(Xc) < 10:
            print(f"[WARN] Skipping regression for '{out}' (insufficient samples).")
            continue

        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            Xc, yc, test_size=0.2, random_state=SEED
        )
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_raw)
        Xte = scaler.transform(Xte_raw)

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
    # 5.7  PCA – PC1 vs PC2 colored by each outcome (optional)
    # ------------------------------------------------------------------ #
    if args.pca:
        for out in outcomes:
            Xc, yc = prepare_xy_for_outcome(out)
            if len(Xc) < 3 or Xc.shape[1] < 2:
                print(f"[WARN] Skipping PCA for '{out}' (need ≥3 samples & ≥2 features).")
                continue

            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xc.values)

            pca = PCA(n_components=2, svd_solver='full')
            Z = pca.fit_transform(Xs)
            ev = pca.explained_variance_ratio_
            pc1, pc2 = Z[:, 0], Z[:, 1]

            plt.figure(figsize=(9, 7))
            if is_classification_target(yc):
                classes = np.unique(yc)
                for cls in classes:
                    mask = (yc.values == cls)
                    plt.scatter(pc1[mask], pc2[mask], label=str(cls), alpha=0.85, s=28)
                plt.legend(title=out, loc='best', fontsize=10)
            else:
                sc = plt.scatter(pc1, pc2, c=yc.values, alpha=0.85, s=28, cmap='viridis')
                cbar = plt.colorbar(sc); cbar.set_label(out)

            ttl = f"PCA by {out}\nExplained variance: PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%"
            plt.title(ttl, pad=14)
            plt.xlabel("PC1"); plt.ylabel("PC2")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output, f"pca_{normalize_name(out)}.pdf"))
            plt.close()

    # ------------------------------------------------------------------ #
    # 5.8  FIGURE A – CONFUSION MATRICES
    # ------------------------------------------------------------------ #
    def grid_shape(n):
        if n <= 4:   return (1, n)
        if n <= 6:   return (2, 3)
        if n <= 9:   return (3, 3)
        return (3, 4)

    for out in classification_out:
        if out not in conf_mats or not conf_mats[out]:
            continue
        nr, nc = grid_shape(len(meth_cls))
        fig, axes = plt.subplots(nr, nc, figsize=(4.4*nc, 3.8*nr))
        axes = as_1d_axes(axes)
        for i, m in enumerate(meth_cls):
            if m not in conf_mats[out]:
                axes[i].axis("off"); continue
            sns.heatmap(conf_mats[out][m], annot=True, fmt='g',
                        cmap=plt.cm.Blues, ax=axes[i])
            axes[i].set_title(m, fontsize=18)
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
        for ax in axes[len(meth_cls):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, f"pdfA_{normalize_name(out)}.pdf"))
        plt.close()

    # ------------------------------------------------------------------ #
    # 5.9  FIGURE B – IMPORTANCE BAR GRIDS
    # ------------------------------------------------------------------ #
    for out in classification_out:
        nr, nc = grid_shape(len(meth_cls))
        fig, axes = plt.subplots(nr, nc, figsize=(5.5*nc, 7*nr))
        axes = as_1d_axes(axes)
        for i, m in enumerate(meth_cls):
            imp_vec = importances[out].get(m, None)
            if imp_vec is None or len(imp_vec) == 0:
                axes[i].axis("off"); continue
            order = np.argsort(imp_vec)[::-1]
            axes[i].barh(np.array(feature_cols)[order], imp_vec[order], color='steelblue')
            axes[i].invert_yaxis()
            axes[i].tick_params(axis='y', labelsize=11)
            axes[i].set_title(m, fontsize=18)
            axes[i].set_xlabel("Mean Decrease")
        for ax in axes[len(meth_cls):]:
            ax.axis("off")
        fig.subplots_adjust(left=0.38)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, f"pdfB_{normalize_name(out)}.pdf"), bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------ #
    # 5.10  FIGURE 3 – Best model per categorical outcome
    # ------------------------------------------------------------------ #
    if len(classification_out) > 0 and not recall_df.empty:
        ordered_panels = classification_out
        best_method = {o: recall_df.loc[o, meth_cls].idxmax()
                       for o in ordered_panels if o in recall_df.index}

        nr, nc = grid_shape(len(best_method))
        w3 = args.width_fig3  if args.width_fig3  is not None else 18
        h3 = args.height_fig3 if args.height_fig3 is not None else 14
        fig3, axes3 = plt.subplots(nr, nc, figsize=(w3, h3))
        axes3 = as_1d_axes(axes3)

        for idx, out in enumerate(best_method.keys()):
            model_name = best_method[out]
            imp_vec = importances[out].get(model_name, None)
            if imp_vec is None or len(imp_vec) == 0:
                axes3[idx].axis("off"); continue
            order = np.argsort(imp_vec)[::-1]
            axes3[idx].barh(np.array(feature_cols)[order], imp_vec[order], color='steelblue')
            axes3[idx].invert_yaxis()
            axes3[idx].tick_params(axis='y', labelsize=10)
            pretty_out = out.replace('_', ' ').title()
            axes3[idx].set_title(f"({chr(65+idx)}) {pretty_out} – {model_name}", fontsize=18)
            axes3[idx].set_xlabel("Mean Decrease")

        for ax in axes3[len(best_method):]:
            ax.axis("off")

        fig3.subplots_adjust(left=0.38)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "Fig3_paper.pdf"), bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------ #
    # 5.11  IMPORTANCES DOT-PLOT (bigger dots; user-tunable size)
    # ------------------------------------------------------------------ #
    all_methods = ([m for m in meth_cls if any(m in importances[o] for o in outcomes)] +
                   [m for m in meth_reg if any(m in importances[o] for o in outcomes)])
    if all_methods:
        valid_arrays = [arr for d in importances.values() for arr in d.values()
                        if arr is not None and len(arr) > 0 and np.isfinite(arr).all()]
        if valid_arrays:
            vals = np.abs(np.concatenate(valid_arrays))
            p95 = np.percentile(vals, 95.0) if np.any(vals > 0) else 1.0
            scale = float(args.imp_dot_scale)
            smin  = float(args.imp_dot_min)
            gmin = float(np.min(vals)) if np.isfinite(vals).all() else 0.0
            gmax = float(np.max(vals)) if np.isfinite(vals).all() else 1.0
            if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
                gmin, gmax = 0.0, 1.0
            norm = plt.Normalize(gmin, gmax); cmap = plt.cm.inferno

            nr, nc = grid_shape(len(all_methods))
            fig_imp, axes_imp = plt.subplots(nr, nc, figsize=(10*nc, 9*nr))
            axes_imp = as_1d_axes(axes_imp)

            for j, m in enumerate(all_methods):
                mat = np.zeros((len(outcomes), len(feature_cols)))
                for r, out in enumerate(outcomes):
                    if m in importances[out]:
                        vec = importances[out][m]
                        if vec is not None and len(vec) == len(feature_cols):
                            mat[r, :] = vec
                for r in range(mat.shape[0]):
                    for c in range(mat.shape[1]):
                        val = mat[r, c]
                        size = smin + scale * (abs(val) / (p95 + 1e-9))
                        axes_imp[j].scatter(c, r, s=size, color=cmap(norm(abs(val))))
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
            fig_imp.subplots_adjust(wspace=0.45, hspace=0.75, bottom=0.28, right=0.9)
            plt.savefig(os.path.join(args.output, "importances.pdf"))
            plt.close()

    # ------------------------------------------------------------------ #
    # 5.12  RECALL HEAT-MAP (Fig 2)
    # ------------------------------------------------------------------ #
    if len(classification_out) > 0 and not recall_df.empty:
        w2 = args.width_fig2  if args.width_fig2  is not None else (1.4*len(meth_cls))
        h2 = args.height_fig2 if args.height_fig2 is not None else (1.2*len(classification_out))
        plt.figure(figsize=(w2, h2))
        sub = recall_df.loc[classification_out]
        sns.heatmap(sub, annot=True, fmt=".2f", cmap="inferno")
        plt.title("Recall Scores (macro)", pad=18)
        plt.xlabel("Methods"); plt.ylabel("Outcomes")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "Fig2_paper.pdf"))
        plt.close()

    # ------------------------------------------------------------------ #
    # 5.13  REGRESSION METRICS TXT
    # ------------------------------------------------------------------ #
    if len(continuous_out) > 0 and reg_metrics:
        with open(os.path.join(args.output, "regression_metrics_summary.txt"), "w") as f:
            f.write("Regression Metrics (MSE | RMSE | MAE | R2)\n\n")
            for out in continuous_out:
                if out not in reg_metrics: continue
                f.write(f"{out}:\n")
                for m in meth_reg:
                    if m not in reg_metrics[out]: continue
                    mse, rmse, mae, r2 = reg_metrics[out][m]
                    f.write(f"  {m:<10}  MSE={mse:.2f}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.2f}\n")
                f.write("\n")

    # ------------------------------------------------------------------ #
    # 5.14  FEATURE SUBSETS CSV EXPORT
    # ------------------------------------------------------------------ #
    subset_union = {"02": set(), "50": set(), "25": set()}

    def best_cls(out): return recall_df.loc[out].idxmax()
    def best_reg(out): return min(reg_metrics[out],
                                  key=lambda m: reg_metrics[out][m][0])

    for out in outcomes:
        if out in classification_out and out in recall_df.index and len(meth_cls) > 0:
            m = best_cls(out)
        elif out in continuous_out and out in reg_metrics and len(meth_reg) > 0:
            m = best_reg(out)
        else:
            continue
        imp_vec = importances[out].get(m, None)
        if imp_vec is None or len(imp_vec) == 0:
            continue
        order = np.argsort(imp_vec)[::-1]
        mask = imp_vec > 0.02
        if np.any(mask):
            subset_union["02"].update(np.array(feature_cols)[mask])
        else:
            subset_union["02"].update([feature_cols[order[0]]])
        subset_union["50"].update(np.array(feature_cols)[order[:max(1, int(len(order)*0.5))]])
        subset_union["25"].update(np.array(feature_cols)[order[:max(1, int(len(order)*0.25))]])

    for tag, feats in subset_union.items():
        keep = [c for c in list(feats) + outcomes if c in data.columns]
        if len(keep) == 0:
            continue
        data.loc[:, keep].to_csv(os.path.join(args.output, f"subset_{tag}.csv"), sep=';', index=True)

    print(f"\n[INFO] All outputs written to: {args.output}\n")

###############################################################################
if __name__ == "__main__":
    main()

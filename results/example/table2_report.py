#!/usr/bin/env python3
# table2_report.py : reproduce “Table 2” with proper statistical tests
import argparse, math, warnings, pandas as pd, numpy as np
from pathlib import Path
import scipy.stats as ss
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------ #
# 0.  Cochran–Armitage trend with fallback -------------------------- #
# ------------------------------------------------------------------ #
try:
    from statsmodels.stats.contingency_tables import cochran_armitage
except Exception:                         # statsmodels < 0.14
    print("[INFO] using built‑in Cochran–Armitage function")

    def cochran_armitage(table, scores=None):
        import math
        table = np.asarray(table, dtype=float)
        if table.shape[0] != 2:
            raise ValueError("table must be 2×k")
        if scores is None:
            scores = np.arange(table.shape[1], dtype=float)
        else:
            scores = np.asarray(scores, dtype=float)

        n   = table.sum()
        p1  = table[1].sum() / n
        q   = table.sum(axis=0) / n
        s   = scores
        num = np.dot(s, table[1]) - n * p1 * np.dot(s, q)
        var = p1 * (1 - p1) * n * np.dot(q, (s - np.dot(s, q))**2)

        z   = num / np.sqrt(var) if var > 0 else np.nan
        p   = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return z, p

# ------------------------------------------------------------------ #
# 1.  CLI ----------------------------------------------------------- #
# ------------------------------------------------------------------ #
ap = argparse.ArgumentParser()
ap.add_argument("--csv",      required=True, help="Input dataframe")
ap.add_argument("--out_csv",  default="Table2_results.csv")
ap.add_argument("--out_pdf",  default="Table2_report.pdf")
args = ap.parse_args()

# ------------------------------------------------------------------ #
# 2.  Smart CSV reader  -------------------------------------------- #
# ------------------------------------------------------------------ #
def smart_read(path: Path) -> pd.DataFrame:
    for d in [',', ';', '\t']:
        df = pd.read_csv(path, delimiter=d)
        if df.shape[1] > 1:
            return df
    raise ValueError("could not detect delimiter")

df = smart_read(args.csv)

# ------------------------------------------------------------------ #
# 3.  Helper functions --------------------------------------------- #
# ------------------------------------------------------------------ #
def median_iqr(series):
    q1, q2, q3 = series.quantile([.25, .5, .75])
    return f"{q2:.0f} ({q1:.0f}-{q3:.0f})"

def mean_sd(series, digits=2):
    return f"{series.mean():.{digits}f} ± {series.std():.{digits}f}"

def percent_n(series):
    n = series.sum()
    N = series.count()
    return f"{100*n/N:.1f} ({int(n)}/{int(N)})"

def p_stars(p):
    return ("****" if p < 1e-4 else
            "***"  if p < 1e-3 else
            "**"   if p < 1e-2 else
            "*"    if p < 5e-2 else "")

def chi2_or_fisher(table):
    # use Fisher if any expected < 5
    chi2, p, _, exp = ss.chi2_contingency(table, correction=False)
    return p if (exp >= 5).all() else ss.fisher_exact(table)[1]

# ------------------------------------------------------------------ #
# 4.  Define rows exactly as in the manuscript --------------------- #
# ------------------------------------------------------------------ #
ROWS = [
    # section,         outcome, predictor,       type,      unit/label
    ("Live newborn vs Stillbirth", "newborn_vital_status", "n_pregnancies", "ordinal", "Number of pregnancies"),
    ("",                          "newborn_vital_status", "primigravidity", "binary",  "Primigravidity"),

    ("Without malformations vs With malformations", "newborn_malformations", "maternal_age", "numeric", "Maternal age"),
    ("",                                   "newborn_malformations", "socioeconomic_level", "ordinal", "Socioeconomic status"),
    ("",                                   "newborn_malformations", "n_pregnancies", "ordinal", "Number of pregnancies"),

    ("Without IUGR vs With IUGR", "iugr", "iugr_history_personal_family", "binary", "Personal or family history of IUGR"),
    ("",                         "iugr", "preterm_birth_history_family", "binary", "Family history of preterm birth"),

    ("Cesarean vs Vaginal delivery", "delivery_type", "maternal_age", "numeric", "Maternal age"),
    ("",                            "delivery_type", "bmi",          "numeric", "BMI"),
]

# outcome coding must be 0 / 1
OUTCOME_LABELS = {
    "newborn_vital_status": (0, 1),
    "newborn_malformations": (0, 1),
    "iugr": (0, 1),
    "delivery_type": (0, 1),
}

# ------------------------------------------------------------------ #
# 5.  Compute statistics and p‑values ------------------------------ #
# ------------------------------------------------------------------ #
records = []
for section, y, x, kind, label in ROWS:
    if y not in df.columns or x not in df.columns:
        print(f"[WARN] '{y}' or '{x}' missing – skipped"); continue

    a, b = OUTCOME_LABELS[y]
    grp0 = df.loc[df[y] == a, x].dropna()
    grp1 = df.loc[df[y] == b, x].dropna()

    if kind == "ordinal":
        # Cochran–Armitage
        levels = sorted(df[x].dropna().unique())
        table = pd.crosstab(df[y], pd.Categorical(df[x], categories=levels,
                                                  ordered=True),
                            dropna=False).reindex(index=[a, b],
                                                  columns=levels, fill_value=0)
        Z, p = cochran_armitage(table.values, scores=levels)
        # Display each level as its own row (like the manuscript)
        for lv in levels:
            pct0 = 100 * (df[(df[y]==a) & (df[x]==lv)].shape[0]) / max(1, grp0.shape[0])
            pct1 = 100 * (df[(df[y]==b) & (df[x]==lv)].shape[0]) / max(1, grp1.shape[0])
            txt0 = f"{pct0:.3g} ({df[(df[y]==a) & (df[x]==lv)].shape[0]}/{grp0.shape[0]})"
            txt1 = f"{pct1:.3g} ({df[(df[y]==b) & (df[x]==lv)].shape[0]}/{grp1.shape[0]})"
            star = p_stars(p)
            records.append([section if lv==levels[0] else "", f"{lv}", "%", txt0, txt1,
                            f"{p:.4g}" if lv==levels[0] else "", star if lv==levels[0] else ""])
            section = ""  # only once
    elif kind == "binary":
        table = pd.crosstab(df[y], df[x]).reindex(index=[a, b], columns=[0, 1],
                                                  fill_value=0)
        p = chi2_or_fisher(table.values)
        txt0 = percent_n(grp0)
        txt1 = percent_n(grp1)
        records.append([section, label, "%", txt0, txt1, f"{p:.4g}", p_stars(p)])
    else:  # numeric
        p = ss.mannwhitneyu(grp0, grp1, alternative="two-sided").pvalue
        txt0 = median_iqr(grp0)
        txt1 = median_iqr(grp1)
        unit = "Years" if x == "maternal_age" else ("kg/m²" if x == "bmi" else "")
        records.append([section, label, unit, txt0, txt1, f"{p:.4g}", p_stars(p)])

# ------------------------------------------------------------------ #
# 6.  Build DataFrame & save CSV ----------------------------------- #
# ------------------------------------------------------------------ #
tabl = pd.DataFrame(records,
                    columns=["Section", "Risk factor", "Unit",
                             "Group 0", "Group 1", "p‑value", "Sig"])
tabl.to_csv(args.out_csv, index=False)
print(f"[INFO] CSV saved → {args.out_csv}")

# ------------------------------------------------------------------ #
# 7.  Render PDF table with matplotlib ----------------------------- #
# ------------------------------------------------------------------ #
plt.rcParams["font.size"] = 9
fig, ax = plt.subplots(figsize=(8.5, 7.5))
ax.axis("off")

# add a big bold title
ax.text(0.5, 1.02, "Table 2. Statistically significant risk factors for pregnancy outcomes other than PE.",
        ha="center", va="bottom", fontsize=12, fontweight="bold", transform=ax.transAxes)

# build cell texts
cell_text, cell_colours = [], []
for _, row in tabl.iterrows():
    if row["Section"]:
        # section header row
        cell_text.append([row["Section"], "", "", "", "", "", ""])
        cell_colours.append(["#f0f0f0"]*7)
    subrow = [row["Risk factor"], row["Unit"],
              row["Group 0"], row["Group 1"],
              row["p‑value"], row["Sig"], ""]
    cell_text.append(subrow)
    cell_colours.append(["white"]*7)

# create table
the_table = ax.table(cellText=cell_text,
                     colLabels=["Risk factor", "Unit",
                                "Group 0", "Group 1", "p‑value", "", ""],
                     cellLoc="center", colLoc="center",
                     colWidths=[0.26, 0.07, 0.18, 0.18, 0.08, 0.04, 0.02],
                     loc="upper left")

the_table.auto_set_font_size(False)
the_table.set_fontsize(8)
the_table.scale(1, 1.4)

plt.savefig(args.out_pdf, bbox_inches="tight")
print(f"[INFO] PDF saved → {args.out_pdf}")

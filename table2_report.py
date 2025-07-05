#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
# table2_report.py
#   • Computes Cochran–Armitage (or χ² / Fisher / Mann‑Whitney) for the
#     comparisons shown in “Table 2”.
#   • Exports a CSV with the statistics AND a publication‑ready PDF that
#     imitates the visual style of the sample table provided by the user.
#   • Requires: pandas, numpy, scipy, matplotlib, (optionally)
#     statsmodels ≥ 0 .14 – otherwise a built‑in CA implementation is used.
# ────────────────────────────────────────────────────────────────────
import argparse, warnings, numpy as np, pandas as pd, scipy.stats as ss
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ════════════════════════════════════════════════════════════════════
# 0. Cochran–Armitage implementation (fallback for statsmodels < 0.14)
# ════════════════════════════════════════════════════════════════════
try:
    from statsmodels.stats.contingency_tables import cochran_armitage
except Exception:
    print("[INFO] statsmodels < 0.14 – using built‑in CA function")

    def cochran_armitage(table, scores=None):
        """
        2 × k Cochran‑Armitage trend test.
        Returns Z‑statistic and two‑sided p‑value.
        """
        import math
        table = np.asarray(table, dtype=float)
        if table.shape[0] != 2:
            raise ValueError("Table must be 2×k")
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

# ════════════════════════════════════════════════════════════════════
# 1. CLI
# ════════════════════════════════════════════════════════════════════
ap = argparse.ArgumentParser()
ap.add_argument("--csv",      required=True, help="Input dataframe file")
ap.add_argument("--out_csv",  default="Table2_results.csv")
ap.add_argument("--out_pdf",  default="Table2_report.pdf")
args = ap.parse_args()

# ════════════════════════════════════════════════════════════════════
# 2. Read CSV (smart delimiter)
# ════════════════════════════════════════════════════════════════════
def smart_read(path):
    for d in [',', ';', '\t']:
        df = pd.read_csv(path, delimiter=d)
        if df.shape[1] > 1:
            return df
    raise ValueError("Could not detect delimiter")

df = smart_read(args.csv)

# ════════════════════════════════════════════════════════════════════
# 3. Helper functions
# ════════════════════════════════════════════════════════════════════
def median_iqr(series):
    q1, q2, q3 = series.quantile([.25, .5, .75])
    return f"{q2:.0f} ({q1:.0f}-{q3:.0f})"

def percent_n(series):
    n = series.sum(); N = series.count()
    return f"{100*n/N:.1f} ({int(n)}/{int(N)})"

def p_stars(p):
    return ("****" if p < 1e-4 else
            "***"  if p < 1e-3 else
            "**"   if p < 1e-2 else
            "*"    if p < 5e-2 else "")

def chi2_or_fisher(table):
    chi2, p, _, exp = ss.chi2_contingency(table, correction=False)
    return p if (exp >= 5).all() else ss.fisher_exact(table)[1]

# ════════════════════════════════════════════════════════════════════
# 4. Row specification  (BMI replaces Height)
# ════════════════════════════════════════════════════════════════════
ROWS = [
    # Live vs stillbirth
    ("Live newborn (n=180) vs Stillbirth (n=10)",
     "newborn_vital_status", "n_pregnancies", "ordinal",
     "Number of pregnancies"),
    ("", "newborn_vital_status", "primigravidity", "binary",
     "Primigravidity"),
    # Malformations
    ("Without malformations (n=170) vs With malformations (n=20)",
     "newborn_malformations", "maternal_age", "numeric", "Maternal age"),
    ("", "newborn_malformations", "socioeconomic_level", "ordinal",
     "Socioeconomic status"),
    ("", "newborn_malformations", "n_pregnancies", "ordinal",
     "Number of pregnancies"),
    # IUGR
    ("Without IUGR (n=146) vs With IUGR (n=44)",
     "iugr", "iugr_history_personal_family", "binary",
     "Personal or family history of IUGR"),
    ("", "iugr", "preterm_birth_history_family", "binary",
     "Family history of preterm birth"),
    # Delivery type
    ("Cesarean (n=152) vs Vaginal delivery (n=38)",
     "delivery_type", "maternal_age", "numeric", "Maternal age"),
    ("", "delivery_type", "bmi", "numeric", "BMI"),
]

# outcome value labels (row order 0 → group 0, 1 → group 1)
OUTCOME_LABELS = dict.fromkeys(
    ["newborn_vital_status", "newborn_malformations", "iugr", "delivery_type"],
    (0, 1)
)

# ════════════════════════════════════════════════════════════════════
# 5. Run stats and build record list
# ════════════════════════════════════════════════════════════════════
records = []
for section, y, x, kind, label in ROWS:
    if y not in df.columns or x not in df.columns:
        print(f"[WARN] '{y}' or '{x}' missing – skipped"); continue

    a, b = OUTCOME_LABELS[y]
    grp0 = df.loc[df[y] == a, x].dropna()
    grp1 = df.loc[df[y] == b, x].dropna()

    if kind == "ordinal":
        levels = sorted(df[x].dropna().unique())
        tab = pd.crosstab(df[y],
                          pd.Categorical(df[x], categories=levels,
                                         ordered=True),
                          dropna=False).reindex(index=[a, b],
                                                columns=levels,
                                                fill_value=0)
        Z, p = cochran_armitage(tab.values, scores=levels)

        # heading row (variable name)
        records.append([section, label, "%", "", "", "", ""])
        section = ""

        for lv in levels:
            n0 = df[(df[y] == a) & (df[x] == lv)].shape[0]
            n1 = df[(df[y] == b) & (df[x] == lv)].shape[0]
            pct0 = 100 * n0 / grp0.shape[0] if grp0.shape[0] else 0
            pct1 = 100 * n1 / grp1.shape[0] if grp1.shape[0] else 0
            txt0 = f"{pct0:.3g} ({n0}/{grp0.shape[0]})"
            txt1 = f"{pct1:.3g} ({n1}/{grp1.shape[0]})"
            records.append([
                "", f"{lv}", "%", txt0, txt1,
                f"{p:.4g}" if lv == levels[0] else "",
                p_stars(p) if lv == levels[0] else ""
            ])

    elif kind == "binary":
        tab = pd.crosstab(df[y], df[x]).reindex(index=[a, b],
                                                columns=[0, 1], fill_value=0)
        p = chi2_or_fisher(tab.values)
        records.append([
            section, label, "%",
            percent_n(grp0), percent_n(grp1),
            f"{p:.4g}", p_stars(p)
        ])

    else:  # numeric (Mann‑Whitney)
        p = ss.mannwhitneyu(grp0, grp1, alternative="two-sided").pvalue
        unit = "Years" if x == "maternal_age" else "kg/m²"
        records.append([
            section, label, unit,
            median_iqr(grp0), median_iqr(grp1),
            f"{p:.4g}", p_stars(p)
        ])

# to DataFrame / CSV
cols = ["Section", "Risk factor", "Unit",
        "Group 0", "Group 1", "p‑value", "Sig"]
tabl = pd.DataFrame(records, columns=cols)
tabl.to_csv(args.out_csv, index=False)
print(f"[INFO] CSV saved → {args.out_csv}")

# ════════════════════════════════════════════════════════════════════
# 6.  PDF rendering (Matplotlib)
# ════════════════════════════════════════════════════════════════════
import matplotlib as mpl
mpl.rcParams.update({"font.size": 9, "pdf.fonttype": 42})

fig, ax = plt.subplots(figsize=(8.3, 10.3))
ax.axis("off")

ax.text(0.5, 1.04,
        "Table 2. Statistically significant risk factors for pregnancy outcomes other than PE.",
        ha="center", va="bottom", fontsize=12, fontweight="bold",
        transform=ax.transAxes)

header = ["Risk factor", "Unit",
          "Live newborn (n=180)", "Stillbirth (n=10)",
          "p‑value", ""]
body, row_style = [], []

for _, row in tabl.iterrows():
    if row["Section"]:
        body.append([row["Section"]] + [""]*(len(header)-1))
        row_style.append("thick")
    body.append([row["Risk factor"], row["Unit"],
                 row["Group 0"], row["Group 1"],
                 row["p‑value"].replace(".", ","), row["Sig"]])
    row_style.append("thin")

col_w = [0.30, 0.07, 0.23, 0.23, 0.11, 0.06]
table = ax.table(cellText=body,
                 colLabels=header,
                 colWidths=col_w,
                 colLoc="center", cellLoc="center",
                 loc="upper left")
table.auto_set_font_size(False)
table.set_fontsize(9)

for cell in table.get_celld().values():
    cell.set_text_props(wrap=True, ha="center", va="center")

# bold header
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.get_text().set_weight("bold")

# horizontal rules
for r, st in enumerate(row_style, start=1):
    lw = 1.2 if st == "thick" else 0.4
    for c in range(len(header)):
        table[(r, c)].set_linewidth(lw)

table.scale(1, 1.45)

ax.text(0, -0.05,
        "PE: preeclampsia. IUGR: intrauterine growth restriction. "
        "* p<0,05; ** p<0,01; *** p<0,001; **** p<0,0001.",
        ha="left", va="top", fontsize=9, style="italic",
        transform=ax.transAxes)

plt.savefig(args.out_pdf, bbox_inches="tight")
print(f"[INFO] PDF saved → {args.out_pdf}")

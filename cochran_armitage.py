#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# cochran_armitage.py
#   Cochran–Armitage trend‑test p‑values for ordinal variables
#   (education_level and socioeconomic_level).
#
#   Inputs
#   ------
#   --csv        : main dataset (CSV / TSV – delimiter auto‑detected)
#   --meta_xlsx  : metadata sheet with columns
#                  variable_name | variable_type | variable_use | table | note
#
#   Outputs
#   -------
#   Table1_results.csv / Table2_results.csv
#   Table1_report.pdf  / Table2_report.pdf   (first column ≈ 30 % width)
#
#   Dependencies
#   ------------
#   pandas, numpy, scipy, matplotlib, openpyxl
#   (statsmodels optional – if < 0.14, a built‑in CA fallback is used)
# ─────────────────────────────────────────────────────────────────────────────

import argparse, warnings, numpy as np, pandas as pd, scipy.stats as ss
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Cochran–Armitage (with fallback) ────────────────────────────────────────
try:
    from statsmodels.stats.contingency_tables import cochran_armitage
except Exception:
    print("[INFO] statsmodels < 0.14 – using built‑in exact CA function")

    def cochran_armitage(tab, scores=None):
        """
        2 × k Cochran–Armitage trend test (exact, two‑sided).
        Returns Z‑statistic and p‑value.
        """
        import math
        tab = np.asarray(tab, float)
        if tab.shape[0] != 2:
            raise ValueError("Input must be a 2 × k table")
        k = tab.shape[1]
        scores = np.asarray(scores if scores is not None else np.arange(k), float)
        if scores.size != k:
            raise ValueError("scores length mismatch")

        n   = tab.sum()
        p1  = tab[1].sum() / n
        q   = tab.sum(0) / n
        s   = scores
        num = (s * tab[1]).sum() - n * p1 * (s * q).sum()
        var = p1 * (1 - p1) * n * ((q * (s - (s*q).sum())**2).sum())
        z   = num / np.sqrt(var) if var > 0 else np.nan
        p   = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return z, p

# ── helpers ─────────────────────────────────────────────────────────────────
def stars(p):
    return "****" if p < 1e-4 else "***" if p < 1e-3 else \
           "**"   if p < 1e-2 else "*"  if p < .05  else ""

def pct_n(part, total):
    return f"{100*part/total:.3g} ({part}/{total})"

# smart CSV/TSV reader -------------------------------------------------------
def smart_read(path: str) -> pd.DataFrame:
    for delim in [",", ";", "\t", "|"]:
        df = pd.read_csv(path, delimiter=delim)
        if df.shape[1] > 1:
            return df
    raise ValueError("Could not detect a valid delimiter for the data file")

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="Main dataframe")
parser.add_argument("--meta_xlsx", required=True, help="Variable metadata XLSX")
args = parser.parse_args()

df   = smart_read(args.csv)
meta = pd.read_excel(args.meta_xlsx, engine="openpyxl")
meta["table"] = meta["table"].astype(str).fillna("")

# masks for Table 1 (EOP vs LOP) & Table 2 (Live vs Still)
is_tbl1 = meta["table"].str.contains("1", regex=False)
is_tbl2 = meta["table"].str.contains("2", regex=False)

# group masks
EOP = df["preeclampsia_onset"] == 1          # n = 80
LOP = df["preeclampsia_onset"] == 0          # n = 110
LIVE  = df["newborn_vital_status"] == 0      # n = 180
DEAD  = df["newborn_vital_status"] == 1      # n = 10

# ── Table 1: EOP vs LOP ─────────────────────────────────────────────────────
print("\n────────  TABLE 1 (EOP vs LOP) ────────")
t1_records = []
for var in ("education_level", "socioeconomic_level"):
    if var not in df.columns:
        print(f"[WARN] {var} not in dataframe – skipped")
        continue

    levels = sorted(df[var].dropna().unique())
    tab = pd.crosstab(df["preeclampsia_onset"],
                      pd.Categorical(df[var], categories=levels, ordered=True)
                     ).loc[[0, 1], levels].values
    z, p = cochran_armitage(tab, scores=np.arange(1, len(levels)+1))
    p_str = f"{p:.4g}"
    t1_records.append([var.replace("_", " ").title(),  # Risk factor column
                       "%"] +                         # Unit column
                      ["", ""] +                      # Placeholders for EOP/LOP header row
                      [p_str, stars(p)])              # p‑value, Sig

    # add each level row (indent for clarity)
    for lvl in levels:
        eop_ct = df.loc[EOP & (df[var] == lvl)].shape[0]
        lop_ct = df.loc[LOP & (df[var] == lvl)].shape[0]
        eop_txt = pct_n(eop_ct, EOP.sum())
        lop_txt = pct_n(lop_ct, LOP.sum())
        t1_records.append([f"  {lvl}", "%", eop_txt, lop_txt, "", ""])

    print(f"T1 │ {var:<25}  p={p_str}  {stars(p)}")

Table1 = pd.DataFrame(t1_records,
          columns=["Risk factor or secondary outcome", "Unit",
                   "EOP (n=80)", "LOP (n=110)", "P value", "Sig"])
Table1.to_csv("Table1_results.csv", index=False)
print("✓ Table 1 CSV written")

# ── Table 2: Live vs Stillbirth ─────────────────────────────────────────────
print("\n────────  TABLE 2 (Live vs Stillbirth) ────────")
t2_records = []
for var in ("education_level", "socioeconomic_level"):
    if var not in df.columns:
        print(f"[WARN] {var} not in dataframe – skipped")
        continue

    levels = sorted(df[var].dropna().unique())
    tab = pd.crosstab(df["newborn_vital_status"],
                      pd.Categorical(df[var], categories=levels, ordered=True)
                     ).loc[[0, 1], levels].values
    z, p = cochran_armitage(tab, scores=np.arange(1, len(levels)+1))
    p_str = f"{p:.4g}"
    t2_records.append([var.replace("_", " ").title(),
                       "%", "", "", p_str, stars(p)])

    for lvl in levels:
        live_ct = df.loc[LIVE & (df[var] == lvl)].shape[0]
        dead_ct = df.loc[DEAD & (df[var] == lvl)].shape[0]
        live_txt = pct_n(live_ct, LIVE.sum())
        dead_txt = pct_n(dead_ct, DEAD.sum())
        t2_records.append([f"  {lvl}", "%",
                           live_txt, dead_txt, "", ""])

    print(f"T2 │ {var:<25}  p={p_str}  {stars(p)}")

Table2 = pd.DataFrame(t2_records,
          columns=["Risk factor", "Unit",
                   "Live newborn (n=180)", "Stillbirth (n=10)",
                   "p‑value", "Sig"])
Table2.to_csv("Table2_results.csv", index=False)
print("✓ Table 2 CSV written")

# ── PDF rendering helper ───────────────────────────────────────────────────
def to_pdf(df: pd.DataFrame, title: str, pdf_name: str):
    plt.rcParams["pdf.fonttype"] = 42
    rows, cols = df.shape
    fig_h = 0.37*rows + 1.4
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.axis("off")
    ax.set_title(title, loc="left", pad=16,
                 fontsize=12.5, fontweight="bold")
    col_w = [0.30] + [(1-0.30)/(cols-1)]*(cols-1)       # ≈30 % first col
    tbl = ax.table(cellText=df.values,
                   colLabels=df.columns,
                   cellLoc="center",
                   colLoc="center",
                   colWidths=col_w,
                   loc="upper left")

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 1.25)

    # left‑align first column
    for r in range(rows+1):               # +1 for header
        cell = tbl[(r, 0)]
        cell.get_text().set_ha("left")

    # thicker header bottom line
    for c in range(cols):
        tbl[(0, c)].set_linewidth(1.4)

    plt.savefig(pdf_name, bbox_inches="tight")
    plt.close()

to_pdf(Table1,
       "Table 1. Population characteristics and distribution of risk factors "
       "and secondary outcomes",
       "Table1_report.pdf")

to_pdf(Table2,
       "Table 2. Statistically significant risk factors for pregnancy outcomes other than PE.",
       "Table2_report.pdf")

print("✓ PDFs written (Table1_report.pdf, Table2_report.pdf)")

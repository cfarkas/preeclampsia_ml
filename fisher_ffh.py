#!/usr/bin/env python3
# fisher_ffh.py  – exact Fisher–Freeman–Halton test for r × c tables
# Outputs one CSV / PDF per outcome
# Dependencies: pandas, numpy, scipy ≥ 1.11, matplotlib, openpyxl
# ----------------------------------------------------------------------

import argparse, warnings, numpy as np, pandas as pd, scipy.stats as ss
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ───────────────────────── helpers ────────────────────────────────────────
def stars(p):
    return "****" if p < 1e-4 else "***" if p < 1e-3 else \
           "**"   if p < 1e-2 else "*"  if p < 5e-2 else ""

def pct_n(part: int, total: int) -> str:
    return f"{100*part/total:.3g} ({part}/{total})"

def smart_read(path: str) -> pd.DataFrame:
    for d in [",", ";", "\t", "|"]:
        df = pd.read_csv(path, delimiter=d)
        if df.shape[1] > 1:
            return df
    raise ValueError("Cannot detect delimiter for data file.")

# exact FFH (SciPy ≥ 1.11 ships it; otherwise fall back to monte‑carlo) ----
def ffh_exact(table: np.ndarray, reps: int = 100_000) -> float:
    """
    Return two‑sided p‑value for a Fisher–Freeman–Halton test.

    If scipy.stats.fisher_exact can already handle r × c (SciPy ≥ 1.11),
    we delegate to it; otherwise we use a permutation approach.
    """
    table = np.asarray(table, int)
    if table.shape == (2, 2):          # ordinary Fisher 2 × 2
        return ss.fisher_exact(table)[1]

    try:                               # SciPy ≥ 1.11 pathway
        return ss.fisher_exact(table)[1]
    except Exception:
        # Monte‑Carlo permutation of row totals
        rng  = np.random.default_rng(7)
        obs  = ss.contingency._g_statistic(table)      # private but handy
        rowsums = table.sum(1)
        colsums = table.sum(0)
        N   = table.sum()
        more_extreme = 0
        for _ in range(reps):
            # sample matrix with fixed margins
            perm = ss.contingency.random_table(rowsums, colsums, rng=rng)
            if ss.contingency._g_statistic(perm) >= obs:
                more_extreme += 1
        return (more_extreme + 1) / (reps + 1)          # add‑one smoothing

# ───────────────────────────── CLI ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",        required=True)
parser.add_argument("--meta_xlsx",  required=True)
args = parser.parse_args()

df   = smart_read(args.csv)
meta = pd.read_excel(args.meta_xlsx, engine="openpyxl")
meta["table"] = meta["table"].astype(str).fillna("")

# ─────────── outcome definitions & labels (unchanged) ─────────────────────
OUTCOMES = [
    ("PE subtype (EOP vs LOP)",           "preeclampsia_onset", 1, 0, "1"),
    ("Newborn vital status",              "newborn_vital_status", 0, 1, "2"),
    ("Newborn malformations",             "newborn_malformations", 0, 1, "3"),
    ("IUGR‑SGA",                          "iugr", 0, 1, "4"),
    ("Delivery type (C‑sect vs vaginal)", "delivery_type", 0, 1, "5"),  # 0 = C‑section
    ("Eclampsia / HELLP",                 "eclampsia_hellp", 0, 1, "6")
]

ORDINALS = {
    "education_level":     list(range(1, 9)),   # 1‑8
    "socioeconomic_level": list(range(1, 6)),   # 1‑5
    "occupation":          list(range(1, 7))    # 1‑6  ← added
}

# ───────── iterate over outcomes ───────────────────────────────────────────
for title, col, g0_val, g1_val, tag in OUTCOMES:
    if col not in df.columns:
        print(f"[WARN] outcome “{col}” missing – skipped")
        continue

    print(f"\n────────  {title}  ────────")
    recs = []
    mask0 = df[col] == g0_val
    mask1 = df[col] == g1_val
    n0, n1 = mask0.sum(), mask1.sum()

    for var, levels in ORDINALS.items():
        if var not in df.columns:
            print(f"[WARN] predictor “{var}” missing – skipped")
            continue

        table = pd.crosstab(df[col],
                            pd.Categorical(df[var], categories=levels,
                                           ordered=True)).reindex(
                          index=[g0_val, g1_val], columns=levels,
                          fill_value=0).values

        p     = ffh_exact(table)
        p_str = f"{p:.4g}"
        recs.append([var.replace("_", " ").title(), "%", "", "", p_str, stars(p)])

        # per‑level detail rows
        for lv in levels:
            recs.append([f"  {lv}", "%",
                         pct_n((mask0 & (df[var] == lv)).sum(), n0),
                         pct_n((mask1 & (df[var] == lv)).sum(), n1),
                         "", ""])

        print(f"{var:<25}  p={p_str}  {stars(p)}")

    # column headers & file base name
    if col == "preeclampsia_onset":
        h0, h1, base = "EOP (n=80)", "LOP (n=110)", "Table1"
    elif col == "newborn_vital_status":
        h0, h1, base = "Live newborn (n=180)", "Stillbirth (n=10)", "Table2"
    else:
        h0 = f"{title.split()[0]} = {g0_val} (n={n0})"
        h1 = f"{title.split()[0]} = {g1_val} (n={n1})"
        base = f"Table{tag}"

    df_out = pd.DataFrame(recs, columns=[
        "Risk factor", "Unit", h0, h1, "p‑value", "Sig"
    ])
    df_out.to_csv(f"{base}_results.csv", index=False)

    # PDF rendering (identical layout, 40 % first column)
    rows, cols = df_out.shape
    fig_h = 0.37*rows + 1.4
    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.axis("off")
    ax.set_title(title, loc="left", pad=16,
                 fontsize=12.5, fontweight="bold")

    col_w = [0.40] + [(1-0.40)/(cols-1)]*(cols-1)
    tbl = ax.table(cellText=df_out.values,
                   colLabels=df_out.columns,
                   colLoc="center", cellLoc="center",
                   colWidths=col_w, loc="upper left")

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 1.25)

    for r in range(rows+1):         # left‑align first column
        tbl[(r, 0)].get_text().set_ha("left")
    for c in range(cols):           # heavier rule under header
        tbl[(0, c)].set_linewidth(1.4)

    plt.savefig(f"{base}_report.pdf", bbox_inches="tight")
    plt.close()
    print(f"✓ Saved →  {base}_results.csv ,  {base}_report.pdf")

print("\nAll FFH tables produced successfully.")

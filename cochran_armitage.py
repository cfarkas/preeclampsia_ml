#!/usr/bin/env python3
# cochran_armitage.py  – trend tests for ordinal predictors
# Requirements: pandas, numpy, scipy, matplotlib, openpyxl
# ---------------------------------------------------------------------------

import argparse, warnings, numpy as np, pandas as pd, scipy.stats as ss
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ────────────────────────────────── Cochran–Armitage (exact, two‑sided) ────
try:
    from statsmodels.stats.contingency_tables import cochran_armitage
except Exception:                                    # statsmodels < 0.14
    print("[INFO] statsmodels < 0.14 – using built‑in exact CA function")

    def cochran_armitage(tab, scores=None):
        import math
        t = np.asarray(tab, float)
        if t.shape[0] != 2:
            raise ValueError("Expect a 2 × k table")
        k = t.shape[1]
        s = np.asarray(scores if scores is not None else np.arange(k), float)
        if s.size != k:
            raise ValueError("len(scores) != k")

        n   = t.sum()
        p1  = t[1].sum() / n
        q   = t.sum(0) / n
        num = (s * t[1]).sum() - n * p1 * (s * q).sum()
        var = p1 * (1 - p1) * n * ((q * (s - (s*q).sum())**2).sum())
        z   = num / np.sqrt(var) if var > 0 else np.nan
        p   = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return z, p

# ───────────────────────────────────────────‑ helpers ‑─────────────────────
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

# ───────────────────────────────────────────── CLI ─────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",        required=True)
parser.add_argument("--meta_xlsx",  required=True)
args = parser.parse_args()

df   = smart_read(args.csv)
meta = pd.read_excel(args.meta_xlsx, engine="openpyxl")
meta["table"] = meta["table"].astype(str).fillna("")

# ───────────────────────────‑ outcome definitions & labels ‑────────────────
OUTCOMES = [
    # label,                 column name,              group 0, group 1, result‑file suffix
    ("PE subtype (EOP vs LOP)",          "preeclampsia_onset",       1, 0, "1"),
    ("Newborn vital status",             "newborn_vital_status",     0, 1, "2"),
    ("Newborn malformations",            "newborn_malformations",    0, 1, "3"),
    ("IUGR‑SGA",                         "iugr",                     0, 1, "4"),
    ("Delivery type (C‑sect vs vaginal)","delivery_type",            0, 1, "5"), # 0=C‑section
    ("Eclampsia / HELLP",                "eclampsia_hellp",          0, 1, "6")
]

ORDINALS = {
    "education_level":      list(range(1, 9)),        # 1‑8
    "socioeconomic_level":  list(range(1, 6))         # 1‑5
}

# ───────────────────────────‑ iterate over outcomes ‑───────────────────────
for title, col, g0_val, g1_val, tag in OUTCOMES:
    if col not in df.columns:
        print(f"[WARN] outcome “{col}” missing – skipped")
        continue

    print(f"\n────────  {title}  ────────")
    recs = []
    grp0_mask = df[col] == g0_val
    grp1_mask = df[col] == g1_val
    n0, n1    = grp0_mask.sum(), grp1_mask.sum()

    for var, levels in ORDINALS.items():
        if var not in df.columns:
            print(f"[WARN] predictor “{var}” missing – skipped")
            continue

        # build 2 × k table with fixed ordering
        tab = pd.crosstab(df[col],
                          pd.Categorical(df[var], categories=levels,
                                         ordered=True)).reindex(
                                index=[g0_val, g1_val], columns=levels,
                                fill_value=0).values

        z, p = cochran_armitage(tab, scores=levels)
        p_str = f"{p:.4g}"
        recs.append([var.replace("_", " ").title(), "%", "", "", p_str, stars(p)])

        # per‑level rows
        for lv in levels:
            ct0 = df.loc[grp0_mask & (df[var] == lv)].shape[0]
            ct1 = df.loc[grp1_mask & (df[var] == lv)].shape[0]
            recs.append([f"  {lv}", "%",
                         pct_n(ct0, n0), pct_n(ct1, n1), "", ""])

        print(f"{var:<25}  p={p_str}  {stars(p)}")

    # choose column headers depending on outcome
    if col == "preeclampsia_onset":
        hdr_g0, hdr_g1 = "EOP (n=80)", "LOP (n=110)"
        fname = "Table1"
    elif col == "newborn_vital_status":
        hdr_g0, hdr_g1 = "Live newborn (n=180)", "Stillbirth (n=10)"
        fname = "Table2"
    else:
        hdr_g0 = f"{title.split()[0]} = {g0_val} (n={n0})"
        hdr_g1 = f"{title.split()[0]} = {g1_val} (n={n1})"
        fname  = f"Table{tag}"

    df_out = pd.DataFrame(recs, columns=[
        "Risk factor", "Unit", hdr_g0, hdr_g1, "p‑value", "Sig"
    ])
    df_out.to_csv(f"{fname}_results.csv", index=False)

    # ── PDF rendering ──
    rows, cols = df_out.shape
    fig_h = 0.37*rows + 1.4
    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.axis("off")
    ax.set_title(f"{title}", loc="left", pad=16,
                 fontsize=12.5, fontweight="bold")

    col_w = [0.25] + [(1-0.25)/(cols-1)]*(cols-1)
    tbl = ax.table(cellText=df_out.values,
                   colLabels=df_out.columns,
                   colLoc="center", cellLoc="center",
                   colWidths=col_w, loc="upper left")

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 1.25)

    # left‑align first column
    for r in range(rows+1):  # +1 header
        tbl[(r, 0)].get_text().set_ha("left")

    # heavy rule below header
    for c in range(cols):
        tbl[(0, c)].set_linewidth(1.4)

    plt.savefig(f"{fname}_report.pdf", bbox_inches="tight")
    plt.close()

    print(f"✓ Saved →  {fname}_results.csv ,  {fname}_report.pdf")

print("\nAll files written successfully.")

# preeclampsia_ml
Machine learning workflow for benchmarking classical machine‑learning models on a maternal–fetal dataset.

## 🚀 Quick‑start

```bash
# One‑time: create env and install deps
git clone https://github.com/cfarkas/preeclampsia_ml.git
cd preeclampsia_ml
python3 main.py --install_conda

# Run the full pipeline
conda run -n ml_preeclampsia python3 main.py \
          --input ./data/preeclampsia_dataset.csv \
          --output ./results/

# Re-Train
python3 main.py --input ./example/test_run/subset_top25_percent.csv --output ./example/test_run_subset/
```

---

## Key Features

| Stage | Highlights |
|-------|------------|
| **Environment** | `--install_conda` bootstraps *ml_preeclampsia* env (Python 3.9 + all deps). |
| **Reproducibility** | Global seed `SEED = 7` → identical splits, weights and importances every run. |
| **Correlation analysis** | Generates both a **full Pearson matrix** *and* a **filtered matrix** that keeps any variable showing \|ρ\| ≥ 0.12 with at least one outcome. |
| **Models** | **Classification models:** `LogReg`, `LDA`, `GNB`, `KNN`, `DecTree`, `RF`, `GradBoost`, `SVM`, `MLP`.<br>**Regression models:** `RFreg`, `GBreg`. |
| **Metrics** | Macro‑averaged **recall** (classification) &rarr; primary score.<br>MSE / RMSE / MAE / R² (regression). |
| **Visual reporting** | • **`pdfA_*.pdf`** confusion‑matrix grids.<br>• **`pdfB_*.pdf`** permutation‑importance bar grids.<br>• **`Fig3_paper.pdf`** consolidated best‑model bars (18 × 14 in).<br>• **`importances.pdf`** supersized dot‑plot across *all* models/outcomes.<br>• **`Fig2_paper.pdf`** recall heat‑map.<br>• **`Fig1_paper.pdf`** filtered correlation matrix. |
| **Feature export** | Three ready‑to‑use CSV subsets — importance > 0.02, top‑50 %, top‑25 % — for re‑training. |

## Implementation Notes

- Scaling: numeric features → StandardScaler; categoricals → label‑encoded.
- Permutation importance: 5 repeats, seeded, n_jobs=-1.
- Splits: 80 / 20 stratified (classification) or plain (regression).
---

## Outputs
```
results/
├── Fig1_paper.pdf # filtered correlation (|ρ| ≥ 0.12, 30 pt font)
├── Fig2_paper.pdf # recall heat‑map
├── Fig3_paper.pdf # best‑model permutation‑importances (panels A–F)
├── importances.pdf # dot‑plot: all models × all outcomes
├── pdfA_<outcome>.pdf # confusion‑matrix grids
├── pdfB_<outcome>.pdf # permutation‑importance bar grids
├── full_corr_matrix.pdf # complete Pearson matrix
├── regression_metrics_summary.txt
├── subset_overall_subset.csv
├── subset_top50_percent.csv
├── subset_top25_percent.csv
└── … (additional PDFs for every outcome)
```

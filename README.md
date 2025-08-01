# preeclampsia_ml
Machine learning workflow for benchmarking classical machine‑learning models on a maternal–fetal dataset.

## 🚀 Quick‑start

```bash
# One‑time: clone repository, create env and install dependences
git clone https://github.com/cfarkas/preeclampsia_ml.git
cd preeclampsia_ml
python3 main.py --install_conda

# Run the full pipeline
python3 main.py --input dataframe.csv --output ./out_20_80/
# Re-Train
python3 main.py --input ./out_20_80/subset_25.csv --output ./out_20_80/25_perc_subset/

# Run the full pipeline (using k-fold instead of split)
python3 main_kfold.py --input dataframe.csv --output ./out_10kfold/
# Re-Train (using k-fold instead of split)
python3 main_kfold.py --input ./out_10kfold/subset_25.csv --output ./out_10kfold/25_perc_subset/
```
```
# Cochran-Armitage trend test (exact, two‑sided) for ordinal variables
python3 cochran_armitage.py --csv dataframe.csv --meta_xlsx PE_dataset_variables.xlsx

# Exact Fisher–Freeman–Halton test for r × c tables
python fisher_ffh.py --csv dataframe.csv --meta_xlsx PE_dataset_variables.xlsx
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

#### The subsets are obtained by the union of all predictors that fell within the top 25 % (or 50 %) permutation-importance ranks in the best-performing model for any outcome.

## Implementation Notes

- Scaling: numeric features → StandardScaler; categoricals → label‑encoded.
- Permutation importance: 5 repeats, seeded, n_jobs=-1.
- Splits: 80 / 20 stratified (classification) or plain (regression), invoked with ```main.py```
- 10 K-fold cross validation, shuffling variables prior cross-validation, invoked with ```main_kfold.py```
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

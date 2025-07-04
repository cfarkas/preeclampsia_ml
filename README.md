# preeclampsia_ml
machine learning method for prediction outcomes in preeclampsia

### Install & Execution
```
git clone https://github.com/cfarkas/preeclampsia_ml.git
cd preeclampsia_ml

# Help
python3 main.py --help

# Install
python3 main.py --install_conda

# Test Run: Use all data and then re-train with best features. 
python3 main.py --input ./example/dataframe.csv --output ./example/test_run/
python3 main.py --input ./example/test_run/best_features_overall_subset.csv --output ./example/test_run_subset/
```

#### Preeclampsia‑ML Pipeline

Deterministic end‑to‑end **Python / scikit‑learn** workflow for benchmarking a
battery of classical machine‑learning models on maternal–fetal datasets.
Generates publication‑ready figures, performance tables and feature‑subset CSVs
in a single command.

---

## Key Features

| Stage | Highlights |
|-------|------------|
| **Environment** | Optional `--install_conda` flag bootstraps a dedicated **`ml_preeclampsia`** conda env (Python 3.9 + pandas / scikit‑learn / seaborn, etc.). |
| **Reproducibility** | Global seed `SEED = 7` → identical train/test splits, model initialisation and permutation‑importance scores across runs. |
| **Correlation analysis** | *Full* Pearson matrix **AND** a *filtered* matrix (|ρ| ≥ 0.12 to any outcome) |
| **Model zoo** | Classification – `LogReg`, `LDA`, `GNB`, `KNN`, `DecTree`, `RF`, `GradBoost`, `SVM`, `MLP`.<br>Regression – `RFreg`, `GBreg`. |
| **Metrics** | Macro‑averaged **recall** for all classification tasks; MSE / RMSE / MAE / R² for continuous outcomes. |
| **Visual reporting** | *pdfA* confusion‑matrix grids; *pdfB* permutation‑importance bar grids; **Fig 3** – consolidated best‑model bars (18 × 14 in); supersized **`importances.pdf`** dot‑plot across all models/outcomes; **Fig 2** recall heat‑map. |
| **Feature export** | Three ready‑to‑use CSV subsets – overall (importance > 0.02), top‑50 %, top‑25 %. |

---

## Outputs

results/
├── Fig1_paper.pdf # filtered correlation 
├── Fig2_paper.pdf # recall heat‑map
├── Fig3_paper.pdf # best‑model importances (panels A–F)
├── importances.pdf # dot‑plot: all models × outcomes
├── pdfA_<outcome>.pdf # confusion matrix grids
├── pdfB_<outcome>.pdf # permutation‑importance bars
├── full_corr_matrix.pdf
├── regression_metrics_summary.txt
├── subset_overall_subset.csv
├── subset_top50_percent.csv
├── subset_top25_percent.csv
└── … (additional PDFs for every outcome)
 

# preeclampsia_ml
machine learning method for prediction outcomes in preeclampsia

### Install & Execution
```
git clone https://github.com/cfarkas/preeclampsia_ml.git
cd preeclampsia_ml

# Help
python3 ml_pipeline.py --help

# Install
python3 ml_pipeline.py --install_conda

# Run (output in current directory)
python3 ml_pipeline.py --input mydataframe.csv --output ./
```

#### This pipeline systematically tests multiple outcomes from a given dataset as follows:

1) It trains a set of classifiers (Logistic Regression, SVM, Random Forest, MLP, etc.) on each classification outcome.
2) It computes recall (macro‐averaged) as the primary metric.
3) It generates:
- Confusion Matrix PDFs (showing each method’s predictions vs. actual labels).
- Bar Charts of permutation importances (PDF B).
- Radial Plots of permutation importances (PDF C) to visualize which features most influenced each classifier.
- Regression (e.g., RandomForestRegressor, GradientBoostingRegressor, for continuous outcomes like gestational_age_delivery and newborn_weight):

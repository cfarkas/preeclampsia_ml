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

#### This pipeline systematically tests multiple outcomes from a given dataset as follows:

1) It trains a set of classifiers (Logistic Regression, SVM, Random Forest, MLP, etc.) on each classification outcome.
2) It computes recall (macro‐averaged) as the primary metric.
3) It generates:
- Confusion Matrix PDFs (showing each method’s predictions vs. actual labels).
- Bar Charts of permutation importances (PDF B).
- Radial Plots of permutation importances (PDF C) to visualize which features most influenced each classifier.
- Regression (e.g., RandomForestRegressor, GradientBoostingRegressor, for continuous outcomes like gestational_age_delivery and newborn_weight):

4) Will select best features that can be inputted again in the pipeline to benchmark performance with those. 

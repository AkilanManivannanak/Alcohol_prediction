# Alcohol use prediction from personality traits

Predicts whether a person has ever used alcohol from Big Five personality scores, impulsivity, sensation seeking and demographics, using the UCI **Drug Consumption (Quantified)** dataset (1,885 respondents). Three models are compared: a decision tree, XGBoost and a small neural network.

> Course project ("AI Present and Future"). Survey data, not a clinical or screening tool.

## Pipeline (`Alcohol_prediction.py`)

| Step | What happens |
|---|---|
| Data | Downloads the UCI data; any use class above `CL0` (never used) counts as a user |
| Features | Interaction terms (e.g. N×E, O×C), risk scores from impulsivity and sensation seeking, personality aggregates |
| Splits | Stratified 80/20 train/test; 20% of training held out as a validation set |
| Preprocessing | Standard scaling and XGBoost-based feature selection, **fitted on the training split only** |
| Imbalance | SMOTETomek on the training split only (falls back to SMOTE) |
| Models | Decision tree and XGBoost (randomised search, 5-fold CV), Keras MLP with class weights and early stopping |
| Thresholds | Youden's J, **chosen on the validation split**, then applied to the untouched test set |
| Outputs | Metrics in `monitoring/*.json`; plots in `Plots/` (confusion matrices, PR/ROC, SHAP, training curves); models in `models/` |

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn tensorflow shap joblib
python Alcohol_prediction.py
```

## Reading the results

Most respondents have used alcohol, so the classes are very unbalanced and plain accuracy is misleading. Compare balanced metrics (F2, PR-AUC, recall at 90% precision) in `monitoring/` instead.

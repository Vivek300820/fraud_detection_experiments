# AI-Driven Fraud Detection — Experiments Notebook

Colab notebook that trains and evaluates six machine learning models for credit card fraud detection, supporting Chapters 3 and 4 of the dissertation *"AI-Driven Fraud Detection in Banking Using Supervised & Unsupervised ML"*.

## What it does

Runs six models — Logistic Regression, Random Forest, XGBoost, MLP, Isolation Forest, and Autoencoder — on the Kaggle Credit Card Fraud Detection dataset, then evaluates and explains their performance.

## Requirements

- Google Colab (uses `google.colab.files` for upload)
- Dataset: `creditcard.csv` (Kaggle Credit Card Fraud Detection dataset), uploaded when prompted
- Python packages installed automatically by the notebook: `xgboost`, `shap`, `imbalanced-learn` (plus the standard `numpy`, `pandas`, `scikit-learn`, `matplotlib` stack already in Colab)

## How to run

1. Open the notebook in Google Colab.
2. Run the dependency install cell (`!pip install xgboost shap imbalanced-learn`).
3. Run the upload cell and select `creditcard.csv` when prompted.
   - Alternative: mount Google Drive and point `DATA_PATH` at the file instead of uploading each time.
4. Run all remaining cells in order, top to bottom.

## Pipeline

Raw dataset (284,807 transactions) → stratified 60/20/20 train/val/test split → SMOTE oversampling (train set only) → train all six models → evaluate on the held-out test set → SHAP explainability.

- `Time` and `Amount` are standard-scaled; `V1`–`V28` are left as-is since they are already PCA-transformed.
- Evaluation uses the F1-maximising decision threshold per model rather than a fixed 0.5 cut-off.

## Outputs

The notebook generates and saves the following figures (referenced in the dissertation as noted):

| File | Contents | Dissertation reference |
|---|---|---|
| Inline (EDA figure) | Class distribution + transaction amount histogram | Ch. 3.1, Figure 3.1 |
| Inline (pipeline diagram) | End-to-end pipeline flow diagram | Ch. 3.2, Figure 3.2 |
| | Model comparison table (AUC, precision, recall, F1) | Ch. 3.4 / Ch. 4.1, Table 4.1 |
| `fig_roc.png` | ROC curves for all six models | Ch. 4.1, Figure 4.1 |
| `fig_pr.png` | Precision–Recall curves for all six models | Ch. 4.1, Figure 4.2 |
| Inline (confusion matrices) | 2×3 grid of confusion matrices | Ch. 4.1, Figure 4.3 |
| `fig_shap.png` | SHAP summary plot for the best model (XGBoost) | Ch. 4.1, Figure 4.4 |

## Notes

- Random seed (`RANDOM_STATE`) is fixed for reproducibility across all splits and models.
- Results are generated directly from the real dataset none of the figures or metrics are fabricated.

This project uses a tabular medical dataset of hematology and demographic features (e.g., HAEMATOCRIT, HAEMOGLOBINS, ERYTHROCYTE, LEUCOCYTE, THROMBOCYTE, MCH, MCHC, MCV, AGE, SEX) to predict whether a patient’s subsequent treatment status is “in” or “out”. The workflow covers data exploration, preprocessing, model training, class‑imbalance handling, model ensembling, threshold optimization, and rich metric/visualization logging.

Folder structure
CODE.ipynb
Jupyter notebook containing the full training and evaluation pipeline.

DATASET.csv
Raw tabular dataset with hematology features, demographics, and the target column (SOURCE or equivalent; mapped to classes “in” and “out”).

DATASET.docx
Documentation describing the dataset variables and collection protocol (if provided by the data source).

artifacts_patient_treatment/
Automatically generated outputs:

ensemble_pipeline.joblib – trained preprocessing + SMOTE + ensemble model.

label_encoder.joblib – encoder mapping string labels (“in”, “out”) to numeric classes.

metrics.json – all scalar metrics (train/val/test, initial vs optimized threshold, cross‑validation).

metadata.json – experiment configuration (features, model hyperparameters, splits, etc.).

performance_summary.txt – human‑readable summary of key results.

*_split.csv – saved train/val/test splits.

Plots:

class_distribution.png – target balance.

corr_heatmap.png – feature correlation matrix.

hist_*.png – per‑feature histograms.

cm_*.png, cm_*_optimized.png – confusion matrices for each split (before/after threshold tuning).

cv_metrics_comparison.png – cross‑validation accuracy/F1/AUC.

metrics_comparison_all_splits.png – train/val/test metrics side by side.

initial_vs_optimized.png – metrics before vs after threshold optimization.

false_positives_negatives.png – FP/FN counts comparison.

roc_curve_test.png – ROC curve with AUC on test set.

precision_recall_curve_test.png – PR curve with average precision.

sensitivity_specificity.png – sensitivity (recall for “out”) and specificity (recall for “in”).

Methods
1. Data preprocessing
Split into train, validation, and test sets with stratification on the target.

Encode the target labels (“in”, “out”) using a label encoder.

Separate features into numeric (lab values, age) and categorical (sex, etc.).

Numeric pipeline: median imputation + standardization.

Categorical pipeline: most‑frequent imputation + one‑hot encoding.

Combine pipelines via a column transformer to ensure identical preprocessing for all models.

2. Handling class imbalance
Inspect training class distribution and compute counts for each class.

Apply SMOTE oversampling on the preprocessed training data to synthetically balance minority class samples.

Configure XGBoost with scale_pos_weight and RandomForest with class_weight="balanced" to further counter imbalance.

3. Model architecture
The main classifier is a soft‑voting ensemble of three models:

Deep MLP (sklearn MLPClassifier)

Hidden layers: 512 → 256 → 128 → 64 → 32

Activation: ReLU

Regularization: L2 (alpha=5e-3)

Optimization: Adam with adaptive learning rate (learning_rate_init=1e-4)

Early stopping with a validation fraction and patience to prevent overfitting.

XGBoost (XGBClassifier)

Objective: binary logistic.

Hyperparameters: 500 trees, max_depth=4, learning_rate=0.03, subsampling and column subsampling, min_child_weight, gamma, reg_alpha, reg_lambda, and scale_pos_weight tuned for tabular medical data.

Random Forest (RandomForestClassifier)

300 trees, max depth 10, minimum samples per split/leaf, max_features="sqrt", and balanced class weights.

The three models are combined using a soft‑voting classifier with weights [1.5, 2.0, 1.0] to give slightly more influence to XGBoost.

4. Training and evaluation
The full pipeline (preprocessing → SMOTE → ensemble) is trained on the training split.

Initial evaluation on train, val, and test:

Accuracy, precision, recall, F1 score.

ROC AUC from predicted probabilities.

Confusion matrices, sensitivity (recall of positive class), and specificity (recall of negative class).

5‑fold stratified cross‑validation on the combined train+val set to estimate generalization, logging mean and standard deviation of accuracy, F1, and AUC.

5. Threshold optimization
Use validation predicted probabilities to sweep thresholds (e.g., 0.30–0.75).

For each threshold, compute F1, accuracy, sensitivity, and specificity.

Select the threshold that maximizes validation F1 (or a chosen trade‑off).

Re‑evaluate validation and test sets with this optimized threshold, generating new confusion matrices and metrics focused on reducing false positives/negatives.

How to run
Clone the repository and create a virtual environment (optional but recommended).

Install dependencies (example using pip):

bash
pip install scikit-learn xgboost imbalanced-learn matplotlib seaborn numpy pandas joblib
Place the dataset file at DATASET.csv or update the path in the notebook/script.

Open CODE.ipynb and run all cells in order
or convert the notebook logic to a .py script and execute:

bash
python deep_mlp_xgb_ensemble_complete.py
After training completes, inspect the artifacts_patient_treatment/ folder for:

Trained model (ensemble_pipeline.joblib)

Metrics and metadata (metrics.json, metadata.json)

Visualization PNGs and summary text.

Results (example interpretation)
On the held‑out test set, the ensemble typically achieves:

Competitive accuracy and ROC AUC for this small medical tabular dataset.

Balanced sensitivity and specificity after threshold tuning, reducing both false positives and false negatives compared with the default 0.5 cutoff.

Stable cross‑validation metrics, indicating reasonable generalization beyond the train/validation split.

Exact numbers are recorded in metrics.json and summarized in performance_summary.txt.

# Interconnect Customer Churn Prediction

Machine learning application that estimates the probability of customer churn for Interconnect, a telecommunications provider, using contract, billing and service information.

Live Demo: https://telecom-customer-churn-prediction-orrfcukqmuvaby9bncptny.streamlit.app/

## Overview

Interconnect wants to proactively identify customers who are likely to cancel their service ("churn") so that the retention team can offer targeted promotions and personalized plans before those customers leave. This project turns that business need into an end-to-end machine learning pipeline: data integration, exploratory analysis, feature engineering, model comparison, hyperparameter tuning, and a Streamlit application that scores a customer's churn risk in real time.

## Business Problem

Acquiring a new customer is significantly more expensive than retaining an existing one. Without a way to flag at-risk customers, retention efforts are reactive (after cancellation) or untargeted (broad, costly promotions). A churn-risk model lets Interconnect focus retention offers on the customers who most need them, improving retention ROI.

## Objective

Predict the probability that a given customer will churn, using information available about their contract, billing history, personal profile and subscribed services - without using any information that would not be available at prediction time (see [Feature Engineering](#feature-engineering) below).

## Dataset

The data comes from four sources, joined on `customerID`:

| File | Description |
|---|---|
| `contract.csv` | Contract type, billing method, monthly/total charges, contract start/end dates |
| `personal.csv` | Gender, senior citizen status, partner and dependents |
| `internet.csv` | Internet service type and add-ons (security, backup, device protection, tech support, streaming) |
| `phone.csv` | Phone service (multiple lines) |

`contract.csv` contains every customer and is used as the base table; `personal.csv`, `internet.csv` and `phone.csv` are joined onto it with a left join. Customers absent from `internet.csv` or `phone.csv` simply do not have that service, so the corresponding missing values are filled with explicit categories (`"No internet service"`, `"No phone service"`) rather than dropped.

## Project Workflow

1. **Data integration** - merge the four sources on `customerID`.
2. **Data cleaning** - fix data types (`TotalCharges` to numeric, `BeginDate`/`EndDate` to datetime), fill service-related missing values, build the `Churn` target.
3. **Exploratory Data Analysis** - univariate and bivariate analysis of every feature against churn.
4. **Statistical analysis** - hypothesis testing to confirm which differences are statistically significant.
5. **Feature engineering** - build model-ready features while explicitly avoiding data leakage.
6. **Model comparison** - train and compare six algorithms using stratified cross-validation.
7. **Hyperparameter tuning** - `RandomizedSearchCV` optimized directly for ROC-AUC.
8. **Model evaluation** - final, single evaluation on a held-out test set.
9. **Streamlit deployment** - an interactive app that loads the trained pipeline and scores new customers.

## Exploratory Data Analysis

- The dataset contains 7,043 customers with no duplicate rows or duplicate `customerID`s.
- The target is moderately imbalanced: **73.5% active customers vs. 26.5% churned customers**, which motivated a stratified train/validation/test split.
- `TotalCharges` was stored as text and contained 11 blank values, all belonging to customers whose contract started on the dataset's reference date (2020-02-01) - i.e. customers too new to have accumulated any charges yet. These were converted to numeric and filled with `0`.
- No numeric feature (`MonthlyCharges`, `TotalCharges`, tenure, number of services) showed IQR-based outliers worth removing.
- Churned customers tend to have **higher monthly charges**, **lower total accumulated charges** (consistent with shorter tenure), **shorter tenure**, and **fewer contracted services** than customers who stay.
- Categorical variables most associated with churn (by inspection of churn rate per category) include contract type, internet service type, online security, tech support and payment method.

## Statistical Analysis

- **Mann-Whitney U test** (numeric features vs. churn): all four numeric features tested (`MonthlyCharges`, `TotalCharges`, tenure, number of services) showed a statistically significant difference between churned and active customers (p < 0.0001 in every case).
- **Chi-square test of independence** (categorical features vs. churn): every categorical feature tested was significantly associated with churn (p < 0.05) **except `gender`** (p ≈ 0.49, not significant).
- **Cramér's V** (association strength): `Type` (contract type) showed the strongest association with churn (V = 0.410), followed by `OnlineSecurity` (0.347), `TechSupport` (0.343), `InternetService` (0.322) and `PaymentMethod` (0.303). `gender` had a Cramér's V of 0.008, confirming it carries essentially no association with churn.

## Feature Engineering

Two columns directly encode the outcome and were **excluded** from training to avoid data leakage:

- `EndDate` / `EndDateDate` - the cancellation date itself.
- `TenureMonths` (created during EDA) - for churned customers, this was computed using `EndDateDate`, so it indirectly leaks the target and was **not** used in the final model.
- `customerID` was also dropped - it is a unique identifier with no predictive value.

Instead, temporal information was derived exclusively from `BeginDate` (the contract start date, which is known independently of the outcome) relative to the dataset's reference date, `2020-02-01`:

| Feature | Description |
|---|---|
| `CustomerAgeMonths` | Months elapsed between `BeginDate` and the reference date |
| `BeginYear` | Year the contract started |
| `BeginMonth` | Month the contract started |
| `IsMonthToMonth` | 1 if the contract type is "Month-to-month" |
| `AutomaticPayment` | 1 if the payment method is an automatic bank transfer or credit card |
| `SecuritySupportCount` | Count of `OnlineSecurity` + `TechSupport` set to "Yes" |
| `NumServices` | Count of add-on services (security, backup, device protection, tech support, streaming TV/movies, multiple lines) set to "Yes" |
| `HasInternet` / `HasPhone` | Whether the customer has any internet / phone service |

The final model uses 12 numeric and 14 categorical features (26 total). Categorical features are one-hot encoded; numeric features pass through unscaled, since the winning model is tree-based.

> **Reference date note:** the dataset represents a historical snapshot, and `CustomerAgeMonths` (plus `BeginYear`/`BeginMonth`) is computed relative to that snapshot's reference date, `2020-02-01`. The application uses the same fixed reference date at inference time - not the current date - so that every prediction stays consistent with the temporal context the model was trained and evaluated on, avoiding a distribution shift between training and serving. This is a deliberate decision to guarantee reproducibility for this portfolio project, not a limitation to work around; a real production deployment would retrain the model periodically on recent data and advance the reference date accordingly. See `src/feature_engineering.py` for the implementation.

## Machine Learning Models

Seven models were evaluated:

1. Dummy Classifier (`most_frequent`, sanity-check baseline)
2. Logistic Regression
3. Random Forest
4. Extra Trees
5. Gradient Boosting
6. CatBoost
7. LightGBM

All models beyond the baseline were compared using 5-fold stratified cross-validation with **ROC-AUC** as the optimization metric, since ROC-AUC was the project's official evaluation criterion. `RandomizedSearchCV` (`scoring="roc_auc"`) was used to tune Gradient Boosting, Extra Trees, Random Forest, CatBoost and LightGBM.

## Model Selection

Cross-validated / validation-set ROC-AUC after hyperparameter tuning:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **LightGBM** | 0.9056 | 0.8754 | 0.7513 | 0.8086 | **0.9327** |
| Gradient Boosting | 0.8957 | 0.8471 | 0.7406 | 0.7903 | 0.9307 |
| CatBoost | 0.8815 | 0.8416 | 0.6818 | 0.7533 | 0.9223 |
| Random Forest | 0.8694 | 0.8253 | 0.6444 | 0.7237 | 0.9016 |
| Extra Trees | 0.8467 | 0.7548 | 0.6257 | 0.6842 | 0.8806 |
| Dummy Classifier | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

**LightGBM** was selected as the final model because it achieved the highest ROC-AUC on the validation set.

## Final Performance

After selecting LightGBM and its hyperparameters, the model was re-trained on the combined training + validation data and evaluated once on the untouched test set:

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.9453** |
| Accuracy | 0.9099 |
| Precision | 0.9023 |
| Recall | 0.7406 |
| F1-score | 0.8135 |

An ROC-AUC of 0.9453 indicates a very strong ability to rank customers by churn risk. The close agreement between validation (0.9327) and test (0.9453) ROC-AUC indicates the model generalizes well and shows no evident overfitting.

### Final LightGBM hyperparameters

```python
LGBMClassifier(
    n_estimators=997,
    learning_rate=0.0790260231986339,
    num_leaves=13,
    max_depth=8,
    min_child_samples=34,
    subsample=0.9455050034169743,
    colsample_bytree=0.9698397677207304,
    reg_alpha=1.6069369739074215,
    reg_lambda=0.43785559304509875,
    class_weight=None,
    random_state=12345,
)
```

No explicit decision-threshold optimization was performed in the original analysis, since ROC-AUC is threshold-independent and was the project's primary metric. The application and `src/predict.py` therefore use the standard **0.5** probability threshold to turn a predicted probability into a High/Low risk label; this is documented explicitly rather than presented as an optimized value.

## Key Drivers

By LightGBM's native split-based feature importance (the number of times each feature is used to split a tree - `LGBMClassifier`'s default `importance_type="split"`, not a gain-based or SHAP importance), the most influential features are:

1. **TotalCharges** - cumulative amount billed to the customer.
2. **CustomerAgeMonths** - how long the customer has been with Interconnect.
3. **MonthlyCharges** - the customer's current monthly bill.

Contract-timing features (`BeginMonth`), service usage (`NumServices`, `SecuritySupportCount`), internet service type, payment method and streaming add-ons also contribute meaningfully. These importances describe what the model relies on to make predictions, not causal relationships.

## Streamlit Application

The app (`app.py`) lets a user enter a customer's contract, billing, personal and service information and returns:

- an estimated **churn probability**;
- a **High Churn Risk** / **Low Churn Risk** classification (0.5 threshold);
- a short business interpretation of the result;
- an "About the Model" section with the algorithm, metric and real test performance;
- a "Model Insights" section showing the model's top features.

Only variables that cannot be derived automatically are asked of the user; engineered features (tenure, contract-timing, service counts, etc.) are computed internally from the raw inputs, using the exact same logic as training (`src/feature_engineering.py`).

## Repository Structure

```text
interconnect-churn-prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── final_provider/
│       ├── contract.csv
│       ├── personal.csv
│       ├── internet.csv
│       └── phone.csv
│
├── notebooks/
│   └── interconnect_churn_analysis.ipynb
│
├── models/
│   ├── churn_model.pkl
│   └── model_metadata.json
│
└── src/
    ├── __init__.py
    ├── data_processing.py
    ├── feature_engineering.py
    ├── train.py
    └── predict.py
```

## Installation

```bash
git clone <YOUR_REPO_URL>
cd interconnect-churn-prediction

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The repository already ships a trained model (`models/churn_model.pkl`), so `streamlit run app.py` works immediately after installing dependencies - no training step required.

### Reproducing the model from scratch

```bash
python -m src.train
```

This reloads `data/final_provider/*.csv`, rebuilds the features, re-trains the LightGBM pipeline with the notebook's best hyperparameters, evaluates it on the test set, and overwrites `models/churn_model.pkl` and `models/model_metadata.json`.

## Technologies

- Python
- Pandas / NumPy
- Scikit-learn
- LightGBM
- SciPy (statistical tests)
- Matplotlib / Seaborn (EDA visualizations)
- Streamlit
- Joblib (model serialization)

## Author

**Alejandro Cotes**

Data Science student focused on Python, SQL, Machine Learning and Data Analytics.

- LinkedIn: https://www.linkedin.com/in/alejandro-cotes-fornaris/
- GitHub: https://github.com/AC020907

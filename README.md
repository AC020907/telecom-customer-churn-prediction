# Interconnect Customer Churn Prediction

End-to-end Machine Learning project for predicting customer churn in a telecommunications company using contract, billing, personal and service information.

The project covers data integration, exploratory analysis, statistical validation, leakage-aware feature engineering, model comparison, hyperparameter tuning, final evaluation and deployment through Streamlit.

**Tech stack:** Python · Pandas · NumPy · Scikit-learn · LightGBM · CatBoost · SciPy · Matplotlib · Seaborn · Streamlit · Joblib

**Final model:** LightGBM (`LGBMClassifier`)  
**Final test ROC-AUC:** **0.9453**  
**Deployment:** Streamlit application

[Open the live Streamlit application](https://telecom-customer-churn-prediction-orrfcukqmuvaby9bncptny.streamlit.app/)

## Business Problem

Interconnect wants to proactively identify customers who are likely to cancel their service so that the retention team can target promotions and personalized plans before those customers leave.

Without a churn-risk model, retention efforts are reactive or untargeted. A predictive model allows the business to prioritize customers with higher estimated churn risk and use retention resources more efficiently.

## Objective

Predict the probability that a given customer will churn using information available about:

- contract characteristics;
- billing history;
- personal profile;
- internet services;
- phone services;
- subscribed add-ons.

The model is designed to avoid using information that would not be available at prediction time.

## Project Workflow

1. **Data integration** — merge the four raw data sources on `customerID`.
2. **Data cleaning** — correct data types, handle service-related missing values and construct the binary churn target.
3. **Exploratory Data Analysis** — analyze feature distributions and churn patterns.
4. **Statistical analysis** — test whether observed differences and associations are statistically significant.
5. **Feature engineering** — create model-ready features while explicitly preventing target leakage.
6. **Model comparison** — compare six machine-learning algorithms plus a Dummy Classifier baseline.
7. **Hyperparameter tuning** — optimize selected models using `RandomizedSearchCV` with ROC-AUC as the scoring metric.
8. **Model evaluation** — perform a final single evaluation on an untouched test set.
9. **Deployment** — serve the trained model through an interactive Streamlit application.

## Dataset

The data comes from four sources joined on `customerID`:

| File | Description |
|---|---|
| `contract.csv` | Contract type, billing method, monthly/total charges and contract dates |
| `personal.csv` | Gender, senior citizen status, partner and dependents |
| `internet.csv` | Internet service type and internet-related add-ons |
| `phone.csv` | Phone service and multiple-line information |

`contract.csv` contains every customer and is used as the base table.

The remaining tables are merged using left joins. Customers absent from `internet.csv` or `phone.csv` are interpreted as customers without those services rather than as missing observations.

Service-related missing values are therefore replaced with explicit categories such as:

```text
No internet service
No phone service
```

## Exploratory Data Analysis

The final dataset contains **7,043 customers** with:

- no duplicate rows;
- no duplicated `customerID` values;
- a moderately imbalanced target;
- **73.5% active customers**;
- **26.5% churned customers**.

Because the target is imbalanced, stratified splitting is used during model development.

### Data quality findings

`TotalCharges` was originally stored as text and contained **11 blank values**.

Those blank values correspond to customers whose contracts began on the dataset reference date, `2020-02-01`, meaning they were too new to have accumulated charges.

The values were therefore:

1. converted to numeric;
2. filled with `0`.

No IQR-based outliers in the main numeric variables were considered significant enough to justify removing observations.

### Main exploratory patterns

Compared with active customers, churned customers tend to show:

- higher monthly charges;
- lower accumulated total charges;
- shorter customer tenure;
- fewer contracted services.

Categorical variables showing notable differences in churn rate include:

- contract type;
- internet service;
- online security;
- technical support;
- payment method.

## Statistical Analysis

Statistical tests were used to determine whether the patterns observed during EDA were statistically supported.

### Numeric variables

A **Mann-Whitney U test** was applied to the numeric features analyzed against churn.

The following variables showed statistically significant differences between churned and active customers:

- `MonthlyCharges`
- `TotalCharges`
- tenure
- number of services

All reported tests produced:

```text
p < 0.0001
```

### Categorical variables

A **Chi-square test of independence** was applied to categorical variables against churn.

All tested categorical variables showed a statistically significant association with churn at:

```text
p < 0.05
```

except:

```text
gender
```

with approximately:

```text
p ≈ 0.49
```

### Cramér's V

Cramér's V was used to quantify association strength.

The strongest associations found were:

| Variable | Cramér's V |
|---|---:|
| Contract type (`Type`) | 0.410 |
| `OnlineSecurity` | 0.347 |
| `TechSupport` | 0.343 |
| `InternetService` | 0.322 |
| `PaymentMethod` | 0.303 |
| `gender` | 0.008 |

The very low value for `gender` is consistent with the non-significant Chi-square result.

## Feature Engineering

Feature engineering was performed with explicit attention to **target leakage**.

### Leakage prevention

The following variables were excluded from model training:

- `customerID` — unique identifier with no predictive meaning;
- `EndDate` — directly reveals whether a customer cancelled;
- `EndDateDate` — datetime representation of the cancellation date;
- `TenureMonths` — the exploratory version used the cancellation date for churned customers and therefore indirectly leaked the target.

Instead of using the leakage-prone tenure feature, the model uses temporal information derived only from `BeginDate`.

### Fixed reference date

The historical dataset represents a snapshot as of:

```text
2020-02-01
```

The model therefore uses the same fixed reference date during both training and inference.

This prevents a mismatch between the temporal feature distributions used during training and those used during prediction.

The deployed application should therefore be interpreted as answering:

> What would this customer's churn risk have been as of the dataset snapshot date?

A real production system would periodically retrain the model on newer data and advance the reference date accordingly.

### Engineered features

| Feature | Description |
|---|---|
| `CustomerAgeMonths` | Months between `BeginDate` and the reference date |
| `BeginYear` | Contract starting year |
| `BeginMonth` | Contract starting month |
| `IsMonthToMonth` | 1 if the contract is month-to-month |
| `AutomaticPayment` | 1 for automatic bank transfer or credit card payment |
| `SecuritySupportCount` | Count of `OnlineSecurity` and `TechSupport` equal to `"Yes"` |
| `NumServices` | Count of active add-on services |
| `HasInternet` | Whether the customer has internet service |
| `HasPhone` | Whether the customer has phone service |

The final model uses:

- **12 numeric features**
- **14 categorical features**
- **26 total input features**

Categorical variables are encoded using one-hot encoding.

Numeric variables are passed through without scaling because the selected final model is tree-based.

## Machine Learning Models

Seven models were evaluated:

1. Dummy Classifier
2. Logistic Regression
3. Random Forest
4. Extra Trees
5. Gradient Boosting
6. CatBoost
7. LightGBM

The Dummy Classifier uses the `most_frequent` strategy and serves as a sanity-check baseline.

The six machine-learning models were compared using stratified validation, while selected models were tuned with `RandomizedSearchCV`.

The main optimization metric was:

```text
ROC-AUC
```

because ROC-AUC was the primary evaluation criterion for the project.

## Model Selection

Validation-set performance after hyperparameter tuning:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **LightGBM** | 0.9056 | 0.8754 | 0.7513 | 0.8086 | **0.9327** |
| Gradient Boosting | 0.8957 | 0.8471 | 0.7406 | 0.7903 | 0.9307 |
| CatBoost | 0.8815 | 0.8416 | 0.6818 | 0.7533 | 0.9223 |
| Random Forest | 0.8694 | 0.8253 | 0.6444 | 0.7237 | 0.9016 |
| Extra Trees | 0.8467 | 0.7548 | 0.6257 | 0.6842 | 0.8806 |
| Dummy Classifier | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

**LightGBM** was selected as the final model because it achieved the highest validation ROC-AUC.

## Final Performance

After model selection, LightGBM was re-trained using the combined training and validation sets.

The final model was then evaluated once on the untouched test set.

| Metric | Value |
|---|---:|
| **ROC-AUC** | **0.9453** |
| Accuracy | 0.9099 |
| Precision | 0.9023 |
| Recall | 0.7406 |
| F1-score | 0.8135 |

The validation ROC-AUC was approximately:

```text
0.9327
```

and the final test ROC-AUC was approximately:

```text
0.9453
```

The similar validation and test performance suggests stable out-of-sample behavior without a large validation-to-test degradation.

## Final LightGBM Configuration

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

## Decision Threshold

No explicit decision-threshold optimization was performed.

The main optimization metric was ROC-AUC, which is threshold-independent.

The deployed model therefore uses the standard probability threshold:

```text
0.5
```

to convert the predicted churn probability into:

```text
High Churn Risk
Low Churn Risk
```

This threshold should not be interpreted as an optimized business decision rule.

In a production environment, the threshold could instead be selected based on:

- retention campaign cost;
- customer lifetime value;
- intervention budget;
- false-positive cost;
- false-negative cost.

## Key Model Drivers

The project uses LightGBM's native **split-based feature importance**.

This represents the number of times a feature is used to split a tree.

It is:

- not SHAP importance;
- not permutation importance;
- not gain-based importance;
- not a causal interpretation.

The three most frequently used features are:

1. **TotalCharges**
2. **CustomerAgeMonths**
3. **MonthlyCharges**

Other relevant features include:

- `BeginMonth`
- `NumServices`
- `SecuritySupportCount`
- Fiber optic internet service
- Multiple lines
- Payment method
- Online backup
- Streaming services
- Contract type

These importances describe which variables the model relies on for prediction, not causal effects on churn.

## Streamlit Application

The project includes an interactive application implemented in:

```text
app.py
```

The application allows users to enter customer information related to:

- contract;
- billing;
- personal characteristics;
- internet service;
- phone service;
- service add-ons.

The application returns:

- estimated churn probability;
- High / Low churn-risk classification;
- short business interpretation;
- model performance information;
- feature-importance visualization.

Only inputs that cannot be derived automatically are requested from the user.

Engineered features such as:

- `CustomerAgeMonths`
- `BeginYear`
- `BeginMonth`
- `NumServices`
- `HasInternet`
- `HasPhone`
- `IsMonthToMonth`
- `AutomaticPayment`
- `SecuritySupportCount`

are generated internally using the same functions used during training.

### Live application

[Open the Streamlit application](https://telecom-customer-churn-prediction-orrfcukqmuvaby9bncptny.streamlit.app/)

## Training / Inference Consistency

The project separates data preparation, feature engineering, training and inference into reusable modules.

```text
Raw CSV files
   ↓
src/data_processing.py
   ↓
Clean merged dataset
   ↓
src/feature_engineering.py
   ↓
Model-ready features
   ↓
src/train.py
   ↓
Trained LightGBM pipeline
   ↓
models/churn_model.pkl
   ↓
src/predict.py
   ↓
app.py
```

The same feature-engineering logic is reused during both training and inference.

This reduces training-serving skew and helps ensure that the deployed model receives features in the same form as the model used during development.

## Repository Structure

```text
telecom-customer-churn-prediction/
│
├── README.md
├── app.py
├── requirements.txt
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

## Source Code Responsibilities

### `src/data_processing.py`

Handles:

- raw CSV loading;
- table merging;
- data-type corrections;
- churn target construction;
- service-related missing-value handling;
- basic service-usage features.

### `src/feature_engineering.py`

Handles:

- leakage prevention;
- fixed reference date;
- temporal features;
- contract-derived features;
- feature selection;
- categorical preprocessing;
- model feature ordering.

### `src/train.py`

Handles:

- reproducible train / validation / test splits;
- pipeline construction;
- final LightGBM training;
- validation and test evaluation;
- feature-importance extraction;
- model serialization;
- metadata generation.

### `src/predict.py`

Handles:

- model loading;
- input validation;
- inference feature construction;
- churn-probability prediction;
- probability-to-class conversion.

### `app.py`

Provides the Streamlit interface used to score customers interactively.

## Installation

Clone the repository:

```bash
git clone https://github.com/AC020907/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The repository already includes a trained model:

```text
models/churn_model.pkl
```

Therefore, the Streamlit application can be executed immediately after installing the dependencies.

## Reproduce the Model From Scratch

Run:

```bash
python -m src.train
```

The training script:

1. loads the four raw datasets from `data/final_provider/`;
2. reproduces the project's cleaning logic;
3. rebuilds the engineered features;
4. creates the same stratified train / validation / test split;
5. trains the final LightGBM pipeline using the best hyperparameters identified during tuning;
6. evaluates validation performance;
7. re-trains the final model using training + validation data;
8. performs a single evaluation on the untouched test set;
9. saves the trained model and metadata.

The following files are generated or overwritten:

```text
models/churn_model.pkl
models/model_metadata.json
```

## Model Artifacts

### Trained pipeline

```text
models/churn_model.pkl
```

Contains the complete Scikit-learn pipeline:

```text
ColumnTransformer
   ↓
OneHotEncoder / numeric passthrough
   ↓
LightGBM classifier
```

### Model metadata

```text
models/model_metadata.json
```

Stores:

- final model name;
- hyperparameters;
- numeric features;
- categorical features;
- decision threshold;
- validation metrics;
- test metrics;
- feature importance;
- random state.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- CatBoost
- SciPy
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## Main Results

- **Customers analyzed:** 7,043
- **Churn rate:** 26.5%
- **Selected model:** LightGBM
- **Validation ROC-AUC:** 0.9327
- **Test ROC-AUC:** 0.9453
- **Test accuracy:** 0.9099
- **Test precision:** 0.9023
- **Test recall:** 0.7406
- **Test F1-score:** 0.8135
- **Decision threshold:** 0.5
- **Deployment:** Streamlit

## Author

**Alejandro Cotes**

Data Science student focused on Python, SQL, Machine Learning and Data Analytics.

- [LinkedIn](https://www.linkedin.com/in/alejandro-cotes-fornaris/)
- [GitHub](https://github.com/AC020907)

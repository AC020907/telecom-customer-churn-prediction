"""
Interconnect Customer Churn Prediction - Streamlit application.

Machine learning application that estimates the probability of customer
churn based on contract, billing and service characteristics, using the
LightGBM model trained in ``notebooks/interconnect_churn_analysis.ipynb``
(``src/train.py``).

Run locally with:

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.feature_engineering import REFERENCE_DATE
from src.predict import ChurnPredictionError, load_model, predict_churn

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

st.set_page_config(
    page_title="Interconnect Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading trained model...")
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {}


def render_header() -> None:
    st.title("Interconnect Customer Churn Prediction")
    st.caption(
        "Machine learning application that estimates the probability of "
        "customer churn based on contract, billing and service "
        "characteristics."
    )
    st.markdown(
        "Interconnect wants to identify customers who are likely to "
        "cancel their service so that retention offers can be targeted "
        "at the right people. Fill in a customer's information below "
        "and click **Predict Churn Risk** to get an estimate."
    )
    st.caption(
        "Predictions are based on the dataset reference snapshot of "
        "February 1, 2020."
    )


def render_inputs() -> dict:
    st.subheader("Customer Information")

    contract_col, billing_col = st.columns(2)

    with contract_col:
        st.markdown("**Contract Information**")
        begin_date = st.date_input(
            "Contract start date",
            value=REFERENCE_DATE - pd.DateOffset(months=12),
            min_value=REFERENCE_DATE - pd.DateOffset(years=20),
            max_value=REFERENCE_DATE,
            help=(
                "Date the customer's contract began. Must not be later "
                "than the dataset reference snapshot (February 1, 2020) "
                "the model was trained on."
            ),
        )
        contract_type = st.selectbox(
            "Contract type", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with billing_col:
        st.markdown("**Billing Information**")
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=0.5
        )
        total_charges = st.number_input(
            "Total charges to date ($)",
            min_value=0.0,
            max_value=20000.0,
            value=840.0,
            step=1.0,
            help="Cumulative amount billed to the customer so far.",
        )
        st.markdown("**Personal Information**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has a partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])

    st.subheader("Services")
    phone_col, internet_col = st.columns(2)

    with phone_col:
        st.markdown("**Phone**")
        multiple_lines = st.selectbox(
            "Phone service", ["No phone service", "No", "Yes"],
            help="'Yes' means the customer has multiple phone lines.",
        )

    with internet_col:
        st.markdown("**Internet**")
        internet_service = st.selectbox(
            "Internet service", ["No internet service", "DSL", "Fiber optic"]
        )

        has_internet = internet_service != "No internet service"
        internet_option_values = ["Yes", "No"] if has_internet else ["No internet service"]

        add_on_col1, add_on_col2 = st.columns(2)
        with add_on_col1:
            online_security = st.selectbox(
                "Online security", internet_option_values, disabled=not has_internet
            )
            online_backup = st.selectbox(
                "Online backup", internet_option_values, disabled=not has_internet
            )
            device_protection = st.selectbox(
                "Device protection", internet_option_values, disabled=not has_internet
            )
        with add_on_col2:
            tech_support = st.selectbox(
                "Tech support", internet_option_values, disabled=not has_internet
            )
            streaming_tv = st.selectbox(
                "Streaming TV", internet_option_values, disabled=not has_internet
            )
            streaming_movies = st.selectbox(
                "Streaming movies", internet_option_values, disabled=not has_internet
            )

    return {
        "BeginDate": pd.Timestamp(begin_date),
        "Type": contract_type,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "MultipleLines": multiple_lines,
    }


def render_prediction(prediction: int, probability: float, threshold: float) -> None:
    st.subheader("Prediction")

    prob_col, class_col = st.columns(2)
    with prob_col:
        st.metric("Churn Probability", f"{probability * 100:.1f}%")
    with class_col:
        if prediction == 1:
            st.metric("Risk Classification", "High Churn Risk")
        else:
            st.metric("Risk Classification", "Low Churn Risk")

    st.progress(min(max(probability, 0.0), 1.0))

    if prediction == 1:
        st.warning(
            "This customer presents a relatively high probability of "
            "leaving Interconnect. Retention actions such as contract "
            "incentives or personalized offers could be considered."
        )
    else:
        st.success(
            "This customer presents a relatively low probability of "
            "leaving Interconnect based on the information provided."
        )

    st.caption(
        f"Classification uses a {threshold:.2f} probability threshold. "
        "This is an estimated probability from a machine learning model, "
        "not a certainty - it should support, not replace, business "
        "judgment about individual customers."
    )


def render_about_model(metadata: dict) -> None:
    st.subheader("About the Model")

    test_metrics = metadata.get("test_metrics", {})
    roc_auc = test_metrics.get("roc_auc", metadata.get("reference_test_roc_auc_from_notebook", 0.9453))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test ROC-AUC", f"{roc_auc:.3f}")
    if test_metrics:
        col2.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.3f}")
        col3.metric("Precision", f"{test_metrics.get('precision', 0):.3f}")
        col4.metric("Recall", f"{test_metrics.get('recall', 0):.3f}")

    st.markdown(
        """
- **Problem type:** binary classification (will the customer churn or not).
- **Final algorithm:** LightGBM (`LGBMClassifier`), selected out of Logistic
  Regression, Random Forest, Extra Trees, Gradient Boosting and CatBoost
  based on cross-validated ROC-AUC.
- **Primary metric:** ROC-AUC, computed on a held-out test set that was
  never used for training or hyperparameter tuning.
- **Feature groups used:** contract details (type, billing, payment
  method, tenure-derived features), billing amounts, personal
  demographics, and subscribed services (internet, phone and add-ons).
        """
    )


def render_model_insights(metadata: dict) -> None:
    top_features = metadata.get("top_features")
    if not top_features:
        return

    st.subheader("Model Insights")
    st.caption(
        "LightGBM split-based feature importance - the number of times "
        "each feature was used to split a tree, not a measure of gain "
        "or SHAP importance."
    )

    importance_df = pd.DataFrame(top_features).head(15).set_index("feature")
    st.bar_chart(importance_df["importance"])


def main() -> None:
    render_header()
    metadata = get_metadata()

    with st.form("customer_form"):
        customer_data = render_inputs()
        submitted = st.form_submit_button("Predict Churn Risk", type="primary")

    if submitted:
        try:
            model = get_model()
            threshold = metadata.get("decision_threshold", 0.5)
            prediction, probability = predict_churn(
                customer_data, model=model, threshold=threshold
            )
            st.divider()
            render_prediction(prediction, probability, threshold)
        except ChurnPredictionError as exc:
            st.error(f"Invalid customer data: {exc}")
        except FileNotFoundError as exc:
            st.error(str(exc))

    st.divider()
    render_about_model(metadata)
    render_model_insights(metadata)


if __name__ == "__main__":
    main()

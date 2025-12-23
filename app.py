import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Load trained model and feature columns
# -------------------------------------------------
model = joblib.load("churn_model_xgb.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    layout="centered"
)

st.title("📞 Telco Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn based on their details.")

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
tenure = st.number_input(
    "Tenure Months",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-Month", "One Year", "Two Year"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes"]
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes"]
)

phone_service = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

# -------------------------------------------------
# Prepare Input Data (SAFE WAY)
# -------------------------------------------------
input_dict = {
    "Tenure Months": tenure,
    "Monthly Charges": monthly_charges,
    "Contract_One Year": 1 if contract == "One Year" else 0,
    "Contract_Two Year": 1 if contract == "Two Year" else 0,
    "Online Security_Yes": 1 if online_security == "Yes" else 0,
    "Partner_Yes": 1 if partner == "Yes" else 0,
    "Dependents_Yes": 1 if dependents == "Yes" else 0,
    "Tech Support_Yes": 1 if tech_support == "Yes" else 0,
    "Streaming TV_Yes": 1 if streaming_tv == "Yes" else 0,
    "Phone Service_Yes": 1 if phone_service == "Yes" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Align with training feature space
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write("---")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
        st.metric("Churn Probability", f"{probability:.2f}")
    else:
        st.success("✅ Customer is likely to stay")
        st.metric("Churn Probability", f"{probability:.2f}")

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("fraud_model.joblib")
st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.write(
    "Enter transaction details to get a model prediction. This is a demonstration "
    "and should not be used as the sole basis for real financial decisions."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()
if model is None:
    st.error("Model file not found. Run `python train_model.py` first to create fraud_model.joblib.")
    st.stop()

with st.form("transaction_form"):
    amount = st.number_input("Transaction amount", min_value=0.0, value=120.0, step=10.0)
    merchant_id = st.number_input("Merchant ID", min_value=0, value=1001, step=1)
    transaction_type = st.selectbox("Transaction type", ["Online", "POS", "ATM", "Transfer", "Other"])
    location = st.selectbox(
        "Location",
        ["Chicago", "Houston", "San Jose", "Dallas", "San Diego", "New York", "Los Angeles", "Other"]
    )
    date = st.date_input("Transaction date", value=pd.Timestamp.today().date())
    hour = st.slider("Transaction hour", min_value=0, max_value=23, value=12)
    submitted = st.form_submit_button("Check transaction")

if submitted:
    timestamp = pd.Timestamp(date) + pd.Timedelta(hours=hour)
    sample = pd.DataFrame([{
        "Amount": amount,
        "MerchantID": int(merchant_id),
        "TransactionType": transaction_type,
        "Location": location,
        "TransactionDate": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    }])

    # Match the feature engineering used in train_model.py.
    sample["TransactionHour"] = pd.to_datetime(sample["TransactionDate"]).dt.hour
    sample["TransactionDayOfWeek"] = pd.to_datetime(sample["TransactionDate"]).dt.dayofweek
    sample["TransactionMonth"] = pd.to_datetime(sample["TransactionDate"]).dt.month
    sample = sample.drop(columns=["TransactionDate"])

    try:
        prediction = int(model.predict(sample)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(sample)[0][1])

        if prediction == 1:
            st.error("Model result: Potentially fraudulent")
        else:
            st.success("Model result: Not flagged as fraud")

        if probability is not None:
            st.metric("Estimated fraud probability", f"{probability:.1%}")
            st.caption("This probability is model-dependent and is not a guarantee.")
    except Exception as exc:
        st.exception(exc)

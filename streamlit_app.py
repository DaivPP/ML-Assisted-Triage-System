import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page config (futuristic feel)
# ----------------------------
st.set_page_config(
    page_title="ML-Assisted Triage System",
    layout="centered"
)

# ----------------------------
# Load model & scaler
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("triage_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

numeric_features = [
    "age", "heart_rate", "spo2",
    "systolic_bp", "respiratory_rate",
    "temperature"
]

triage_map = {
    0: "Non-Urgent",
    1: "Urgent",
    2: "Critical"
}

# ----------------------------
# UI Header
# ----------------------------
st.markdown(
    "<h1 style='text-align:center; color:#00f5ff;'> ML-Assisted Triage System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI-powered emergency prioritization</p>",
    unsafe_allow_html=True
)

st.divider()

# ----------------------------
# Input Form
# ----------------------------
with st.form("triage_form"):
    st.subheader("🧍 Patient Information")

    age = st.number_input("Age", 0, 100, 45)

    st.subheader("❤️ Vital Signs")
    col1, col2 = st.columns(2)
    with col1:
        heart_rate = st.number_input("Heart Rate (bpm)", 30, 200, 90)
        systolic_bp = st.number_input("Systolic BP (mmHg)", 60, 200, 120)
        temperature = st.number_input("Temperature (°C)", 34.0, 42.0, 36.8)
    with col2:
        spo2 = st.number_input("SpO₂ (%)", 70, 100, 98)
        respiratory_rate = st.number_input("Respiratory Rate (/min)", 8, 40, 16)

    st.subheader("🤒 Symptoms & History")
    col3, col4 = st.columns(2)
    with col3:
        chest_pain = st.checkbox("Chest Pain")
        shortness_of_breath = st.checkbox("Shortness of Breath")
        severe_bleeding = st.checkbox("Severe Bleeding")
    with col4:
        loss_of_consciousness = st.checkbox("Loss of Consciousness")
        diabetes = st.checkbox("Diabetes")
        heart_disease = st.checkbox("Heart Disease")

    submitted = st.form_submit_button("🔍 Analyze Triage")

# ----------------------------
# Prediction
# ----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "systolic_bp": systolic_bp,
        "respiratory_rate": respiratory_rate,
        "temperature": temperature,
        "chest_pain": int(chest_pain),
        "shortness_of_breath": int(shortness_of_breath),
        "severe_bleeding": int(severe_bleeding),
        "loss_of_consciousness": int(loss_of_consciousness),
        "diabetes": int(diabetes),
        "heart_disease": int(heart_disease)
    }])

    input_data[numeric_features] = scaler.transform(
        input_data[numeric_features]
    )

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities)

    triage_level = triage_map[prediction]

    st.divider()

    if triage_level == "Critical":
        st.error(f"🚨 CRITICAL CASE DETECTED ({confidence*100:.1f}% confidence)")
        st.markdown("**Immediate drone dispatch & hospital alert recommended**")
    elif triage_level == "Urgent":
        st.warning(f"⚠️ URGENT CASE ({confidence*100:.1f}% confidence)")
        st.markdown("Doctor consultation required")
    else:
        st.success(f"✅ NON-URGENT CASE ({confidence*100:.1f}% confidence)")
        st.markdown("Teleconsultation is sufficient")


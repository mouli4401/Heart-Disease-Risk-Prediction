# app.py
import streamlit as st
import pandas as pd
import pickle

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page config
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="💓", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Adjust inputs to predict the 10-year heart disease risk.")

# Main Title
st.markdown("<h1 style='text-align: center; color: red;'>💓 Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter patient details below to estimate their 10-year risk of CHD.</p>", unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 40)
    education = st.selectbox("Education Level", [1, 2, 3, 4])
    sex = st.selectbox("Sex", ["M", "F"])
    is_smoking = st.selectbox("Smoking", ["YES", "NO"])
    cigsPerDay = st.number_input("Cigs per Day", 0, 60, 0)
    BPMeds = st.selectbox("On BP meds", [0, 1])
    prevalentStroke = st.selectbox("History of Stroke", [0, 1])

with col2:
    prevalentHyp = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    totChol = st.number_input("Total Cholesterol", 100, 400, 200)
    sysBP = st.number_input("Systolic BP", 90, 250, 120)
    diaBP = st.number_input("Diastolic BP", 60, 150, 80)
    BMI = st.number_input("BMI", 15.0, 50.0, 25.0)
    heartRate = st.number_input("Heart Rate", 40, 150, 70)
    glucose = st.number_input("Glucose", 50, 300, 80)

# Convert categorical
sex = 1 if sex == "M" else 0
is_smoking = 1 if is_smoking == "YES" else 0

# Prepare input
input_data = pd.DataFrame([[age, education, sex, is_smoking, cigsPerDay, BPMeds, prevalentStroke,
                            prevalentHyp, diabetes, totChol, sysBP, diaBP,
                            BMI, heartRate, glucose]],
                          columns=["age","education","sex","is_smoking","cigsPerDay","BPMeds","prevalentStroke",
                                   "prevalentHyp","diabetes","totChol","sysBP","diaBP",
                                   "BMI","heartRate","glucose"])

# Scale
input_scaled = scaler.transform(input_data)

# Predict
if st.button("🔍 Predict Risk", use_container_width=True):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ **High risk of heart disease**\n\nEstimated Probability: **{prob:.2f}**")
    else:
        st.success(f"✅ **Low risk of heart disease**\n\nEstimated Probability: **{prob:.2f}**")

    # Show input summary
    st.markdown("### 📊 Patient Summary")
    st.dataframe(input_data.style.highlight_max(axis=1))

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HealthGuard – Disease Prediction", layout="centered")

st.title("🧠 HealthGuard – AI-Based Disease Prediction System")
st.markdown("### Predict the likelihood of **Diabetes** using Machine Learning")

# Load the trained model
model = joblib.load("src/model.pkl")

# User input fields
st.sidebar.header("🧍‍♀️ Patient Health Information")

def user_input():
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.number_input("Glucose", 0, 200, 100)
    bloodpressure = st.sidebar.number_input("Blood Pressure", 0, 130, 70)
    skinthickness = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
    bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.number_input("Age", 1, 100, 33)
    
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bloodpressure,
        "SkinThickness": skinthickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

st.subheader("🧩 Patient Input Data")
st.write(input_df)

if st.button("🔍 Predict"):
    result = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if result == 1:
        st.error(f"⚠️ High Risk of Diabetes Detected (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Diabetes (Confidence: {prob:.2f})")

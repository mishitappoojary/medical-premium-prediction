import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the models
with open('rf_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('stacking_model.pkl', 'rb') as stacking_file:
    stacking_model = pickle.load(stacking_file)

# Function to predict using both models
def predict(input_data):
    rf_prediction = rf_model.predict(input_data)
    stacking_prediction = stacking_model.predict(input_data)

    return rf_prediction, stacking_prediction

# Streamlit App Layout
st.title("Medical Preminum Prediction")

# Input fields for user info
age = st.number_input("Age", min_value=0, max_value=120, value=25)
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

diabetes = st.radio("Diabetes (0 = No, 1 = Yes)", [0, 1])
blood_pressure_problems = st.radio("Blood Pressure Problems (0 = No, 1 = Yes)", [0, 1])
any_transplants = st.radio("Any Transplants (0 = No, 1 = Yes)", [0, 1])
any_chronic_diseases = st.radio("Any Chronic Diseases (0 = No, 1 = Yes)", [0, 1])
known_allergies = st.radio("Known Allergies (0 = No, 1 = Yes)", [0, 1])
history_of_cancer = st.radio("History of Cancer in Family (0 = No, 1 = Yes)", [0, 1])
number_of_major_surgeries = st.number_input("Number of Major Surgeries", min_value=0, value=2)

# Create DataFrame from user input
user_input = pd.DataFrame({
    'Age': [age],
    'Diabetes': [diabetes],
    'BloodPressureProblems': [blood_pressure_problems],
    'AnyTransplants': [any_transplants],
    'AnyChronicDiseases': [any_chronic_diseases],
    'Height': [height],
    'Weight': [weight],
    'KnownAllergies': [known_allergies],
    'HistoryOfCancerInFamily': [history_of_cancer],
    'NumberOfMajorSurgeries': [number_of_major_surgeries]
})

# Predict and display results when the user clicks the button
if st.button("Predict"):
    rf_pred, stacking_pred = predict(user_input)

    st.subheader("Predictions")
    st.success(f"Random Forest Prediction: {rf_pred[0]}")
    st.success(f"Stacking Model Probability: {stacking_pred[0]:.4f}")


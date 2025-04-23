
import streamlit as st
import pickle
import pandas as pd

# Load model
with open('disease_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

st.title("Disease Prediction from Symptoms")

symptoms = ['Itching', 'Skin Rash', 'Nodal Skin Eruptions', 'Continuous Sneezing', 'Shivering',
            'Chills', 'Joint Pain', 'Stomach Pain', 'Acidity', 'Ulcers On Tongue', 'Muscle Wasting',
            'Vomiting', 'Burning Micturition', 'Spotting Urination', 'Fatigue', 'Weight Gain', 'Anxiety']

user_input = {}
for symptom in symptoms:
    user_input[symptom] = st.selectbox(f"Do you have {symptom}?", ['No', 'Yes'])

# Convert to dataframe
input_df = pd.DataFrame([user_input])
for col in input_df.columns:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Predict
if st.button("Predict Disease"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Disease: {prediction[0]}")

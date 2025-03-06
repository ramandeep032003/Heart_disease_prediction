import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load pre-trained model and scaler from the .pkl files

import os

# Function to load the model using pickle with error handling
import pickle
import os

# Function to load the model and scaler
def load_model_and_scaler(model_filename, scaler_filename):
    model = None
    scaler = None
    
    # Load model if the file exists
    if os.path.exists(model_filename):  
        try:
            with open(model_filename, 'rb') as model_file:
                model = pickle.load(model_file)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading the model: {e}")
    
    # Load scaler if the file exists
    if os.path.exists(scaler_filename):
        try:
            with open(scaler_filename, 'rb') as scaler_file:
                scaler = pickle.load(scaler_file)
            print("Scaler loaded successfully!")
        except Exception as e:
            print(f"Error loading the scaler: {e}")
    
    if model and scaler:
        return model, scaler
    else:
        print("Either model or scaler could not be loaded.")
        return None, None

# Example of loading the model and scaler
model_filename = 'lr (1).pkl'  # Path to your model file
scaler_filename = 'sc (1).pkl'  # Path to your scaler file

model, scaler = load_model_and_scaler(model_filename, scaler_filename)

# If the model and scaler are loaded successfully, you can use them here
if model and scaler:
    print("Model and Scaler are ready for predictions.")
    # For example, you can now use the model and scaler for predictions
else:
    print("Failed to load model or scaler.")

# Streamlit UI
st.title("Heart Disease Prediction")

st.write("This app predicts whether a person has heart disease based on various health parameters.")

# User input
age = st.slider("Age", 29, 77, 50)
sex = st.selectbox("Sex", ['Male', 'Female'])
sex = 1 if sex == 'Male' else 0  # Male=1, Female=0
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1])
thalach = st.slider("Maximum Heart Rate Achieved", 60, 200, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("Depression Induced by Exercise", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2,3])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3,4])
thal = st.selectbox("Thalassemia", [0,1,2,3])

# Create input array from user input
input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1,-1)
input_data_scaled = scaler.transform(input_data)


# Predict using the loaded model
prediction = model.predict(input_data_scaled)

# Display the result
if prediction >= 0.5:  # You can adjust the threshold as needed
    st.write("Prediction: The person has heart disease.")
else:
    st.write("Prediction: The person does not have heart disease.")
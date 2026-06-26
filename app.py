import streamlit as st
import pickle
import pandas as pd
import os

# Get the absolute directory path where app.py is sitting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build bulletproof paths to your files
model_path = os.path.join(BASE_DIR, "safetour_model.pkl")
encoder_path = os.path.join(BASE_DIR, "weather_encoder.pkl")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load encoder
with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

st.title("SafeTour AI Safety Predictor")

crime = st.slider("Crime History", 0, 10)

weather = st.selectbox(
    "Weather Condition",
    ["Sunny", "Cloudy", "Rainy", "Stormy"]
)

crowd = st.slider("Crowd Density", 0, 100)

emergency = st.slider(
    "Emergency Services Nearby",
    0,
    10
)

if st.button("Predict Safety Score"):

    weather_encoded = encoder.transform([weather])[0]

    data = pd.DataFrame([[
        crime,
        weather_encoded,
        crowd,
        emergency
    ]], columns=[
        "crime_history",
        "weather_condition",
        "crowd_density",
        "emergency_services_nearby"
    ])

    prediction = model.predict(data)[0]

    st.success(
        f"Predicted Safety Score: {prediction:.2f}/100"
    )

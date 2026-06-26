import streamlit as st
import pickle
import pandas as pd

# Load model
with open("safetour_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoder
with open("weather_encoder.pkl", "rb") as f:
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
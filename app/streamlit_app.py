import streamlit as st
import joblib
import numpy as np

# Load the trained model and feature list
model = joblib.load("../modeling_cancellation_prediction.joblib")
features = joblib.load("../PredictionModelFeature.joblib")

st.set_page_config(page_title="Hotel Booking Cancellation Predictor")
st.title("🏨 Hotel Booking Cancellation Predictor")

st.write("Enter the booking details below to predict whether the booking will be cancelled:")

# Collect input values
lead_time = st.number_input("Lead Time (in days)", min_value=0, max_value=700, value=100)
adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
is_repeated_guest = st.selectbox("Is Repeated Guest?", [0, 1])

# Construct input feature vector
input_data = [lead_time, adults, is_repeated_guest]

# Pad the remaining features with default zeroes
while len(input_data) < len(features):
    input_data.append(0)

# Convert to numpy array
input_array = np.array(input_data).reshape(1, -1)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_array)[0]
    if prediction == 1:
        st.error("❌ This booking is likely to be CANCELLED.")
    else:
        st.success("✅ This booking is likely to be HONOURED.")

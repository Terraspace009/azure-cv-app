import streamlit as st
import joblib
import numpy as np
import os

# Load the trained model
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "..", "models", "modeling_cancellation_prediction.joblib")
model = joblib.load(model_path)

# Streamlit UI setup
st.set_page_config(page_title="Hotel Booking Cancellation Predictor")
st.title("📅 Hotel Booking Cancellation Predictor")
st.write("Enter the booking details below to predict whether the booking will be cancelled:")

# Collect input values
lead_time = st.number_input("Lead Time (in days)", min_value=0, max_value=700, value=100)
adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
is_repeated_guest = st.selectbox("Is Repeated Guest?", [0, 1])

# Build input feature vector
input_data = [lead_time, adults, is_repeated_guest]

# Pad with zeroes to match model input size
while len(input_data) < model.n_features_in_:
    input_data.append(0)

# Convert to numpy and reshape
input_array = np.array(input_data).reshape(1, -1)

# Predict and show result
if st.button("Predict"):
    prediction = model.predict(input_array)[0]
    result = "🔴 Will Cancel" if prediction == 1 else "🟢 Will Not Cancel"
    st.success(f"Prediction: {result}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained models and scaler
svr_model = joblib.load('best_svr_model.pkl')  # Assuming you saved the trained model
xgb_model = joblib.load('xgb_model.pkl')
meta_model = joblib.load('meta_model_svr_xgboost.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler that was used for training

st.title('Traffic Volume Prediction')

# Collect user input
junction = st.selectbox('Select Junction', [1, 2, 3, 4], help="Select the junction number")
hour = st.slider('Hour of the Day', 0, 23, help="Select the hour in 24-hour format")
day = st.slider('Day of the Month', 1, 31, help="Select the day of the month")
month = st.selectbox('Month', list(range(1, 13)), help="Select the month")
year = st.selectbox('Year', [2023, 2024], help="Select the year")

# Add default values for missing features (holiday and weather)
holiday = 0  # Assuming no holiday (you can change this)
weather = 1  # Assuming normal weather conditions (you can adjust this)

# Prediction logic
if st.button('Predict Traffic Volume'):
    try:
        # Prepare input features for prediction
        input_data = np.array([[junction, hour, day, month, year, holiday, weather]])

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Get predictions from SVM and XGBoost
        y_svr_pred = svr_model.predict(input_scaled)
        y_xgb_pred = xgb_model.predict(input_scaled)

        # Combine predictions for the meta-model (ensemble)
        combined_pred = np.column_stack((y_svr_pred, y_xgb_pred))
        final_pred = meta_model.predict(combined_pred)

        # Debugging information
        st.write("Scaled Input Features:", input_scaled)
        st.write(f"SVM Prediction: {y_svr_pred[0]:.2f} vehicles")
        st.write(f"XGBoost Prediction: {y_xgb_pred[0]:.2f} vehicles")

        # Display the predicted traffic volume
        st.write(f"Predicted Traffic Volume: {final_pred[0]:.2f} vehicles")

    except Exception as e:
        st.error(f"Error in prediction: {e}")

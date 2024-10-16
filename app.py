import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load pre-trained models and scaler
svr_model = joblib.load('best_svr_model.pkl')  # Assuming you saved the trained model
xgb_model = joblib.load('xgb_model.pkl')
meta_model = joblib.load('meta_model_svr_xgboost.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler that was used for training

st.title('Traffic Volume and Density Prediction')

# Collect user input
junction = st.selectbox('Select Junction', [1, 2, 3, 4], help="Select the junction number")
hour = st.slider('Hour of the Day', 0, 23, help="Select the hour in 24-hour format")
day = st.slider('Day of the Month', 1, 31, help="Select the day of the month")
month = st.selectbox('Month', list(range(1, 13)), help="Select the month")
year = st.selectbox('Year', [2023, 2024], help="Select the year")

# Add default values for missing features (holiday and weather)
holiday = 0  # Assuming no holiday (you can change this)
weather = 1  # Assuming normal weather conditions (you can adjust this)

# Define traffic density levels based on predicted traffic volume
def classify_density(traffic_volume):
    if traffic_volume < 5:
        return "Low"
    elif 5 <= traffic_volume < 10:
        return "Moderate"
    elif 10 <= traffic_volume < 20:
        return "High"
    else:
        return "Severe"

# Plot traffic density
def plot_traffic_density(junction, traffic_volume):
    categories = ['Low', 'Moderate', 'High', 'Severe']
    counts = [0, 0, 0, 0]  # To count traffic levels
    
    # Count which level current traffic belongs to
    if traffic_volume < 5:
        counts[0] += 1
    elif 5 <= traffic_volume < 10:
        counts[1] += 1
    elif 10 <= traffic_volume < 20:
        counts[2] += 1
    else:
        counts[3] += 1

    fig, ax = plt.subplots()
    ax.bar(categories, counts, color=['green', 'yellow', 'orange', 'red'])
    ax.set_title(f"Traffic Density at Junction {junction}")
    ax.set_ylabel("Density Level")
    ax.set_xlabel("Traffic Levels")

    st.pyplot(fig)

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

        # Traffic density classification
        traffic_volume = final_pred[0]
        traffic_density = classify_density(traffic_volume)

        # Debugging information
        st.write("Scaled Input Features:", input_scaled)
        st.write(f"SVM Prediction: {y_svr_pred[0]:.2f} vehicles")
        st.write(f"XGBoost Prediction: {y_xgb_pred[0]:.2f} vehicles")

        # Display the predicted traffic volume and density
        st.write(f"Predicted Traffic Volume: {traffic_volume:.2f} vehicles")
        st.write(f"Traffic Density at Junction: **{traffic_density}**")

        # Plot the graphical representation of traffic density
        plot_traffic_density(junction, traffic_volume)

    except Exception as e:
        st.error(f"Error in prediction: {e}")

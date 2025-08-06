import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
from streamlit_lottie import st_lottie
import json

def load_lottie_local(filepath: str):
    with open(filepath, "r") as file:
        return json.load(file)
    
# Function to prepare model
def prepare_model(algorithm, X_train, y_train):
    pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='mean'))])
    model = Pipeline(steps=[('preprocessing', pipeline),('algorithm', algorithm)])
    model.fit(X_train, y_train)
    return model

# Sidebar with sections
st.sidebar.title("EV Battery Prediction App")
section = st.sidebar.selectbox(
    "Select a section:",
    ["Home", "FileUploader", "Predictions", "Model Comparisons", "Reports"]
)

# Home Section
if section == 'Home':
    st.title("Predicting RLU Values for Electric Vehicles")

    lottie_file_path = "Animation - 1726912618065.json"  # Replace with your Lottie file path

    # Load and display the Lottie animation
    lottie_animation = load_lottie_local(lottie_file_path)
    st_lottie(lottie_animation, speed=1, width=700, height=400, key="home_animation")
    
    st.subheader("App Overview")
    
    st.write("""
    This App focuses on predicting the **Remaining Useful Life (RUL)** of electric vehicle batteries using machine learning techniques. 
    Accurate predictions of battery lifespan are crucial for optimizing performance and ensuring user satisfaction in electric vehicles.
    """)

# FileUploader Section
if section == 'FileUploader':
    st.title("Upload Training Data")
    uploaded_file = st.file_uploader("Upload a CSV file for training", type=["csv"])

    if uploaded_file is not None:
        battery_df = pd.read_csv(uploaded_file)
        # Feature Selection (assuming the last column is the target)
        if st.button('Train Model'):
            st.write(battery_df.head())
            target = battery_df['RUL']
            feature = battery_df.drop(['RUL', 'Cycle_Index'], axis=1)
            target.shape, feature.shape
            scaler = StandardScaler()
            feature_std = scaler.fit_transform(feature)
            feature_std = pd.DataFrame(feature_std, columns = feature.columns)
            # Split data into train and test (you can add custom splitting)
            X_train, X_test, y_train, y_test = train_test_split(feature_std, target, test_size=0.2, random_state=2404)

            # Load RandomForestRegressor model
            algorithm = RandomForestRegressor()
            st.write("Training started...")
                
            # Train model
            start_time = time.time()
            model = prepare_model(algorithm, X_train, y_train)
            end_time = time.time()
            #Progression Bar
            progress_bar = st.progress(0)
            for percent_complete in range(99):
                time.sleep(0.05)
                progress_bar.progress(percent_complete + 1)
            st.success("Model Trained Successfully time taken is {}".format(end_time - start_time))
                

# Predictions Section
if section == 'Predictions':
    st.title("Make Predictions")
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Input fields for user to enter values
    discharge_time = st.number_input('Discharge Time (s)', min_value=0)
    decrement_36_34V = st.number_input('Decrement 3.6-3.4V (s)', min_value=0)
    max_voltage_discharge = st.number_input('Max. Voltage Dischar. (V)', min_value=0.0)
    min_voltage_charge = st.number_input('Min. Voltage Charg. (V)', min_value=0.0)
    time_at_415V = st.number_input('Time at 4.15V (s)', min_value=0)
    time_constant_current = st.number_input('Time constant current (s)', min_value=0)
    charging_time = st.number_input('Charging time (s)', min_value=0)

    # Convert the inputs into a dataframe for prediction
    input_data = pd.DataFrame({
        'Discharge Time (s)': [discharge_time],
        'Decrement 3.6-3.4V (s)': [decrement_36_34V],
        'Max. Voltage Dischar. (V)': [max_voltage_discharge],
        'Min. Voltage Charg. (V)': [min_voltage_charge],
        'Time at 4.15V (s)': [time_at_415V],
        'Time constant current (s)': [time_constant_current],
        'Charging time (s)': [charging_time]
    })

    st.write("Data for prediction:")
    st.write(input_data)
    if st.button('Predict'):
        prediction = model.predict(input_data)
        st.success("## Remaining Usefull life of a battery is: {} Hrs".format(prediction))
    

# Model Comparisons Section
if section == 'Model Comparisons':
    st.title("Model Comparisons")
    with open('results_dict.pkl', 'rb') as file:
        results = pickle.load(file)
    df = pd.DataFrame(results)

    # 1. MSE Comparison
    st.subheader("MSE Comparison")
    fig_mse = plt.figure(figsize=(10, 6))
    plt.bar(df['Algorithm'], df['MSE'], color='skyblue')
    plt.title('MSE Comparison')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    st.pyplot(fig_mse)

    # 2. RMSE Comparison
    st.subheader("RMSE Comparison")
    fig_rmse = plt.figure(figsize=(10, 6))
    plt.bar(df['Algorithm'], df['RMSE'], color='orange')
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    st.pyplot(fig_rmse)

    # 3. Time Comparison
    st.subheader("Time Comparison")
    fig_time = plt.figure(figsize=(10, 6))
    plt.bar(df['Algorithm'], df['Time'], color='green')
    plt.title('Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    st.pyplot(fig_time)

    # 4. Accuracy Comparison
    st.subheader("Accuracy Comparison")
    fig_accuracy = plt.figure(figsize=(10, 6))
    plt.bar(df['Algorithm'], df['Accuracy'], color='purple')
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    st.pyplot(fig_accuracy)

# Reports Section
if section == 'Reports':
    st.title("Reports")
    with open('reports.pkl', 'rb') as file:
        reports = pickle.load(file)
    st.write(reports)
    

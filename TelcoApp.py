import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("/Users/bannusagi/.spyder-py3/TelcoChurn.pkl")

# Function to preprocess input data
def preprocess_input(data):
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Convert categorical variables to numeric
    df['gender'] = df['gender'].map({'Female':0, 'Male':1})
    df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes':1, 'No':0})
    df['Partner'] = df['Partner'].map({'Yes':1, 'No':0})
    df['Dependents'] = df['Dependents'].map({'Yes':1, 'No':0})

    df['PhoneService'] = df['PhoneService'].map({'Yes':1, 'No':0})
    df['MultipleLines'] = df['MultipleLines'].map({'No phone service':0, 'Yes':1, 'No':2})

    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'None': 2})
    df['OnlineSecurity'] = df['OnlineSecurity'].map({'No': 0, 'Yes':2, 'No internet service':1})
    df['OnlineBackup'] = df['OnlineBackup'].map({'Yes':2, 'No':0, 'No internet service':1})
    df['DeviceProtection'] = df['DeviceProtection'].map({'Yes':2, 'No':0, 'No internet service':1})
    df['TechSupport'] = df['TechSupport'].map({'Yes':2, 'No':0, 'No internet service':1})
    df['StreamingTV'] = df['StreamingTV'].map({'Yes':2, 'No':0, 'No internet service':1})
    df['StreamingMovies'] = df['StreamingMovies'].map({'Yes':2, 'No':0, 'No internet service':1})

    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes':1, 'No':0})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 2, 'Mailed check': 3, 'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1})
    
    # Return preprocessed DataFrame
    return df

# Streamlit UI
st.title("Customer Churn Prediction")


# Collect user inputs
gender = st.radio("Gender", ['Male', 'Female'])
senior_citizen = st.radio("Senior Citizen", ['Yes', 'No'])
partner = st.radio("Partner", ['Yes', 'No'])
dependents = st.radio("Dependents", ['Yes', 'No'])
phone_service = st.radio("Phone Service", ['No', 'Yes'])
if phone_service == 'Yes':
    multiple_lines = st.radio("Multiple Lines", ['Yes','No'])
else:
    multiple_lines = 'No phone service'
internet_service = st.selectbox("Internet Service", ['None', 'DSL', 'Fiber optic'])
if internet_service != 'None':
    online_security = st.radio("Online Security", ['Yes', 'No'])
    online_backup = st.radio("Online Backup", ['Yes', 'No'])
    device_protection = st.radio("Device Protection", ['Yes', 'No'])
    tech_support = st.radio("Tech Support", ['Yes', 'No'])
    streaming_tv = st.radio("Streaming TV", ['Yes', 'No'])
    streaming_movies = st.radio("Streaming Movies", ['Yes', 'No'])
else:
    online_security = 'No internet service'
    online_backup = 'No internet service'
    device_protection = 'No internet service'
    tech_support = 'No internet service'
    streaming_tv = 'No internet service' 
    streaming_movies = 'No internet service'
    

contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.radio("Paperless Billing", ['Yes', 'No'])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input("Monthly Charges", value=0.0)
total_charges = st.number_input("Total Charges", value=0.0)
tenure_group = st.number_input("Tenure Group", value=0)

# Make prediction
if st.button("Predict"):
    # Create dictionary from user inputs
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure': tenure_group
    }
    
    # Preprocess input data
    processed_data = preprocess_input(user_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Display prediction result
    if prediction[0] == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is likely to stay.")
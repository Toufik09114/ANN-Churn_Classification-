import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle 

# loading the Ann trained model
model = load_model('model.h5')

# loading the encoded gender file
with open('le_gender.pkl', 'rb') as file:
    le_gender = pickle.load(file)

# loading the onehotencoded goegraphy pickle file
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# loading the scaler pickle file
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", le_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_number = st.selectbox("Is Active Number", [0, 1])


# Prepare the data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [le_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_number],
    "EstimatedSalary": [estimated_salary],
})

# encoding Georaphical features
encoded_geo = onehot_encoder_geo.transform([[geography]]).toarray()
encoded_geo_df = pd.DataFrame(encoded_geo, columns = onehot_encoder_geo.get_feature_names_out(["Geography"]))

# Concating the feature
input_data = pd.concat([input_data.reset_index(drop=True), encoded_geo_df], axis='columns')

# Feature Scaling the input data
input_scaled = scaler.transform(input_data)

# Classification button
if st.button("Classify"):
    # Predicting output
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]
    st.write(f"Predicted Probability: {prediction_proba:.2f}")

    if prediction_proba > 0.5:
        st.error("The Customer is likely to churn.")
    else: 
        st.success("The Customer is unlikely to churn.")
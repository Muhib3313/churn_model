import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model and columns
model = pickle.load(open("churn_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("Customer Churn Prediction")

# User inputs
credit_score = st.number_input("Credit Score", 300, 850, 600)
age = st.slider("Age", 18, 100, 40)
balance = st.number_input("Account Balance", 0, 1000000, 50000)
country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])

# Convert to model input
input_dict = {
    'credit_score': credit_score,
    'age': age,
    'balance': balance,
    f"country_{country}": 1,
    f"gender_{gender}": 1
}

# Fill missing columns with 0
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.write("Churn Prediction:", "Yes" if pred == 1 else "No")

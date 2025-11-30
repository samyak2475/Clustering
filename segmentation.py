import streamlit as st
import pandas as pd
import joblib

# Load model & scaler
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation App")
st.write("Enter customer details to predict their segment.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=0, value=20000)
total_spending = st.number_input("Total Spending", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

# Create input
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],   # <— FIXED
    "Recency": [recency]
})

# Order must match scaler training
input_data = input_data[
    ["Age", "Income", "Total_Spending",
     "NumWebPurchases", "NumStorePurchases",
     "NumWebVisitsMonth", "Recency"]
]

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"The predicted customer segment is: Cluster {cluster}")

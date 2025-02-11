#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib

# Load trained models
rf_model_revenue = joblib.load("rf_model_revenue.pkl")
rf_model_spend = joblib.load("rf_model_spend.pkl")
rf_model_profit = joblib.load("rf_model_profit.pkl")

# Streamlit UI
st.title("📈 Revenue Prediction App")
st.write("Enter the number of **billable calls** to predict revenue, spend, and profit.")

# User Input
converted_calls = st.number_input("Enter Billable Calls:", min_value=1, step=1, value=10)

# Prediction Function
def predict(converted_calls):
    input_data = pd.DataFrame([[converted_calls]], columns=['Converted'])
    predicted_revenue = rf_model_revenue.predict(input_data)[0]
    predicted_spend = rf_model_spend.predict(input_data)[0]
    predicted_profit = rf_model_profit.predict(input_data)[0]
    return predicted_revenue, predicted_spend, predicted_profit

# Run Prediction and Display Results
if st.button("Predict"):
    revenue, spend, profit = predict(converted_calls)
    st.success(f"📊 **Predicted Revenue:** ${revenue:,.2f}")
    st.info(f"💰 **Predicted Spend:** ${spend:,.2f}")
    st.warning(f"📈 **Predicted Profit:** ${profit:,.2f}")


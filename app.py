import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request

# Define model filenames
model_files = {
    "rf_model_revenue": "rf_model_revenue.pkl",
    "rf_model_spend": "rf_model_spend.pkl",
    "rf_model_profit": "rf_model_profit.pkl"
}

# Ensure models exist locally; download if missing
for model_name, model_file in model_files.items():
    if not os.path.exists(model_file):
        url = f"https://raw.githubusercontent.com/niksaderek/app/main/{model_file}"
        urllib.request.urlretrieve(url, model_file)

# Load trained models
rf_model_revenue = joblib.load("rf_model_revenue.pkl")
rf_model_spend = joblib.load("rf_model_spend.pkl")
rf_model_profit = joblib.load("rf_model_profit.pkl")

# Streamlit UI with Tableau-like styling
st.set_page_config(page_title="Predictive Model v.06", layout="wide")
st.title("📈 **Predictive Model v.06**")

st.markdown("""
    <style>
        .stButton>button {
            background-color: #3e7c7b;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            height: 50px;
            width: 200px;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }
        .stAlert {
            font-size: 16px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.write("Enter the number of **billable calls** to predict revenue, spend, and profit.")

# User Input with enhanced styling
converted_calls = st.number_input("Enter Billable Calls:", min_value=1, step=1, value=1, format="%d")

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
    
    # Add a small Tableau-inspired data visualization
    st.markdown("""
    <div style="width: 100%; height: 300px; background-color: #f5f5f5; border-radius: 10px; display: flex; justify-content: center; align-items: center; font-size: 24px; font-weight: bold; color: #3e7c7b;">
        Predictive Insights
    </div>
    """, unsafe_allow_html=True)

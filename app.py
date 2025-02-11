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
            background-color: #0076d6; /* Tableau blue */
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

# Calculate weekly revenue, spend, and profit (assuming 5 workdays per week)
def calculate_weekly_values(daily_revenue, daily_spend, daily_profit):
    weekly_revenue = daily_revenue * 5
    weekly_spend = daily_spend * 5
    weekly_profit = daily_profit * 5
    return weekly_revenue, weekly_spend, weekly_profit

# Run Prediction and Display Results
if st.button("Predict"):
    revenue, spend, profit = predict(converted_calls)
    
    st.success(f"📊 **Predicted Revenue (Daily):** ${revenue:,.2f}")
    st.info(f"💰 **Predicted Spend (Daily):** ${spend:,.2f}")
    st.warning(f"📈 **Predicted Profit (Daily):** ${profit:,.2f}")
    
    # Calculate expected weekly values
    weekly_revenue, weekly_spend, weekly_profit = calculate_weekly_values(revenue, spend, profit)
    
    st.success(f"📊 **Expected Weekly Revenue:** ${weekly_revenue:,.2f}")
    st.info(f"💰 **Expected Weekly Spend:** ${weekly_spend:,.2f}")
    st.warning(f"📈 **Expected Weekly Profit:** ${weekly_profit:,.2f}")

    # Create two columns for displaying daily and weekly predictions side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### **Daily Predictions**")
        st.write(f"📊 **Revenue:** ${revenue:,.2f}")
        st.write(f"💰 **Spend:** ${spend:,.2f}")
        st.write(f"📈 **Profit:** ${profit:,.2f}")
    
    with col2:
        st.markdown("### **Weekly Predictions**")
        st.write(f"📊 **Revenue (Week):** ${weekly_revenue:,.2f}")
        st.write(f"💰 **Spend (Week):** ${weekly_spend:,.2f}")
        st.write(f"📈 **Profit (Week):** ${weekly_profit:,.2f}")
    
    # Add a bar chart to visualize the predictions (Daily and Weekly)
    st.markdown("""
    <div style="width: 100%; height: 300px; background-color: #f5f5f5; border-radius: 10px; display: flex; justify-content: center; align-items: center; font-size: 24px; font-weight: bold; color: #0076d6;">
        Predictive Insights
    </div>
    """, unsafe_allow_html=True)
    
    # Create a DataFrame for bar chart visualization
    data = pd.DataFrame({
        "Prediction Type": ["Revenue", "Spend", "Profit"],
        "Daily Values": [revenue, spend, profit],
        "Weekly Values": [weekly_revenue, weekly_spend, weekly_profit]
    })

    # Plot the bar chart using Streamlit's built-in method
    st.bar_chart(data.set_index("Prediction Type"))

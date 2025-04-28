import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from stock_model import  train_model, predict_stock, evaluate_model, plot_predictions
from config import ALPHAVANTAGE_API_KEY

# -----------------------------
# Function to fetch stock data
# -----------------------------
def get_stock_data_from_alpha_vantage(symbol, start_date, end_date):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AMZN&apikey=6HXJV13R2HBYQWPV"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": "AMZN",
        "apikey": ALPHAVANTAGE_API_KEY,
        "outputsize": "full",  # Full historical data
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Handle invalid response
    if 'Time Series (Daily)' not in data:
        st.error("Failed to fetch stock data. Please check the symbol or try again later.")
        return pd.DataFrame(), None

    # Extract metadata
    meta_data = data.get("Meta Data", {})
    time_series = data["Time Series (Daily)"]

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series, orient="index")
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)

    # Filter by date
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    df = df.loc[mask]

    return df, meta_data

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction App")

# Sidebar for inputs
st.sidebar.header("Input Parameters")
stock = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", "AMZN")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Predict Button
if st.sidebar.button("Predict"):
    with st.spinner("Fetching data and building model..."):
        data, meta_data = get_stock_data_from_alpha_vantage(stock, start_date, end_date)

        if not data.empty:
            # Show metadata
            st.subheader("📋 Stock Metadata")
            st.write(f"**Symbol:** {meta_data.get('2. Symbol', 'N/A')}")
            st.write(f"**Last Refreshed:** {meta_data.get('3. Last Refreshed', 'N/A')}")
            st.write(f"**Time Zone:** {meta_data.get('5. Time Zone', 'N/A')}")

            # Show raw data
            st.subheader("📊 Raw Stock Data")
            st.dataframe(data.tail())

            # Model
            model = train_model(data)
            predictions = predict_stock(model, data)

            # Show predictions
            st.subheader("📈 Predicted vs Actual Stock Prices")
            st.plotly_chart(plot_predictions(data, predictions), use_container_width=True)

            # Show metrics
            mae, rmse = evaluate_model(predictions, data)
            st.metric(label="Mean Absolute Error", value=f"{mae:.2f}")
            st.metric(label="Root Mean Squared Error", value=f"{rmse:.2f}")
        else:
            st.error("No data available to train the model.")
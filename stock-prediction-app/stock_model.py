import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go

# --- Load and preprocess the data
def load_data(data):
    close_prices = data['close'].values.reshape(-1, 1)  # Use only 'close' price
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    return scaled_data, scaler

# --- Train the LSTM model
def train_model(data):
    scaled_data, scaler = load_data(data)
    
    X, y = [], []
    time_step = 60  # Use last 60 days to predict the next day

    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # [samples, time_steps, features]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    
    return (model, scaler)

# --- Make predictions
def predict_stock(model_data, data):
    model, scaler = model_data
    scaled_data, _ = load_data(data)

    X_test = []
    time_step = 60

    for i in range(time_step, len(scaled_data)):
        X_test.append(scaled_data[i-time_step:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Convert back to original scale
    return predictions

# --- Evaluate model
def evaluate_model(predictions, data):
    actual = data['close'].values[-len(predictions):]

    mae = np.mean(np.abs(predictions.flatten() - actual))
    rmse = np.sqrt(np.mean((predictions.flatten() - actual)**2))
    return mae, rmse

# --- Plot actual vs predicted prices
def plot_predictions(data, predictions):
    actual = data['close'].values[-len(predictions):]
    dates = data.index[-len(predictions):]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=dates, y=predictions.flatten(), mode='lines', name='Predicted Price'))
    fig.update_layout(title="Stock Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)")
    return fig
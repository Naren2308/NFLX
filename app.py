# app.py

from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load and preprocess data
def load_and_process_data(filename):
    data = pd.read_csv('NFLX.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data[['Close']]
    data['5_MA'] = data['Close'].rolling(window=5).mean()
    data['30_MA'] = data['Close'].rolling(window=30).mean()
    data['Volatility'] = data['Close'].rolling(window=5).std()
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    train_size = int(len(data_scaled) * 0.8)
    train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]
    return train_data, test_data, scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=60, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Load and preprocess data
        data = load_and_process_data('NFLX.csv')
        train_data, test_data, scaler = prepare_data(data)

        lookback = 90
        X_train, y_train = [], []
        X_test, y_test = [], []

        for i in range(lookback, len(train_data)):
            X_train.append(train_data[i-lookback:i])
            y_train.append(train_data[i, 0])
        for i in range(lookback, len(test_data)):
            X_test.append(test_data[i-lookback:i])
            y_test.append(test_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)

        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_predictions = scaler.inverse_transform(np.concatenate((train_predictions, np.zeros((train_predictions.shape[0], 4))), axis=1))[:, 0]
        y_train_actual = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], 4))), axis=1))[:, 0]
        test_predictions = scaler.inverse_transform(np.concatenate((test_predictions, np.zeros((test_predictions.shape[0], 4))), axis=1))[:, 0]
        y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))), axis=1))[:, 0]

        plt.figure(figsize=(14, 6))
        plt.plot(data.index[lookback:len(y_train_actual)+lookback], y_train_actual, color='blue', label='Actual Train Price')
        plt.plot(data.index[lookback:len(train_predictions)+lookback], train_predictions, color='red', label='Predicted Train Price')
        plt.plot(data.index[len(y_train_actual)+2*lookback:], y_test_actual, color='blue', linestyle='dotted', label='Actual Test Price')
        plt.plot(data.index[len(y_train_actual)+2*lookback:], test_predictions, color='green', linestyle='dotted', label='Predicted Test Price')
        plt.title('Netflix Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig('static/plot.png')
        
        return redirect(url_for('result'))

    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html', plot_url='static/plot.png')

if __name__ == '__main__':
    app.run(debug=True)

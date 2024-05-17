from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from pandas_datareader import data as pdr
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.compat.v1.nn import rnn_cell as rnn_cell
from enum import Enum
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

class LTCCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, num_units, input_mapping=MappingType.Affine, solver=ODESolver.SemiImplicit, ode_solver_unfolds=6, activation=tf.nn.tanh, **kwargs):
        super().__init__(**kwargs)
        self._num_units = num_units
        self._ode_solver_unfolds = ode_solver_unfolds
        self._solver = solver
        self._input_mapping = input_mapping
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self._num_units), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self._num_units, self._num_units), initializer='glorot_uniform', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self._num_units,), initializer='zeros', name='bias')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        net_input = tf.matmul(inputs, self.kernel)
        net_input += tf.matmul(prev_output, self.recurrent_kernel)
        net_input += self.bias
        output = self._activation(net_input)  # Use the activation function

        return output, [output]

    def activation(self, net_input):
        pass

    def get_config(self):
        config = super(LTCCell, self).get_config()
        config.update({"num_units": self._num_units})
        return config

class CTRNN(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, global_feedback=False, activation=tf.nn.tanh, cell_clip=None, **kwargs):
        self.units = units
        self.global_feedback = global_feedback
        self.activation = activation
        self.cell_clip = cell_clip
        super(CTRNN, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', name='bias')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        net_input = tf.matmul(inputs, self.kernel)
        net_input += tf.matmul(prev_output, self.recurrent_kernel)
        net_input += self.bias
        output = self.activation(net_input)

        if self.cell_clip is not None:
            output = tf.clip_by_value(output, -self.cell_clip, self.cell_clip)

        return output, [output]

class NODE(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, cell_clip=None, **kwargs):
        self.units = units
        self.cell_clip = cell_clip
        super(NODE, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', name='bias')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        net_input = tf.matmul(inputs, self.kernel)
        net_input += tf.matmul(prev_output, self.recurrent_kernel)
        net_input += self.bias
        output = tf.nn.tanh(net_input)

        if self.cell_clip is not None:
            output = tf.clip_by_value(output, -self.cell_clip, self.cell_clip)

        return output, [output]


class CTGRU(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, cell_clip=None, **kwargs):
        self.units = units
        self.cell_clip = cell_clip
        super(CTGRU, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], 2 * self.units), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, 2 * self.units), initializer='glorot_uniform', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(2 * self.units,), initializer='zeros', name='bias')
        self.kernel_c = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', name='kernel_c')
        self.recurrent_kernel_c = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='recurrent_kernel_c')
        self.bias_c = self.add_weight(shape=(self.units,), initializer='zeros', name='bias_c')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        zr = tf.matmul(inputs, self.kernel)
        zr += tf.matmul(prev_output, self.recurrent_kernel)
        zr += self.bias
        z, r = tf.split(zr, 2, axis=-1)

        z = tf.sigmoid(z)
        r = tf.sigmoid(r)

        c = tf.matmul(inputs, self.kernel_c)
        c += r * tf.matmul(prev_output, self.recurrent_kernel_c)
        c += self.bias_c
        c = tf.nn.tanh(c)

        output = (1 - z) * prev_output + z * c

        if self.cell_clip is not None:
            output = tf.clip_by_value(output, -self.cell_clip, self.cell_clip)

        return output, [output]
    
    


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def feature_engineering(data):
  data.ta.macd(close='Adj Close', fast=14, slow=21, append=True) # 14 & 21 Day MACD
  # Pivot Points (Standard Pivot Levels)
  pivot_data = data[['High', 'Low', 'Adj Close']].copy()

  # We assume that the 'High', 'Low', and 'Close' are from the previous day
  pivot_data['Pivot Point'] = (pivot_data['High'] + pivot_data['Low'] + pivot_data['Adj Close']) / 3
  pivot_data['R1'] = (2 * pivot_data['Pivot Point']) - pivot_data['Low']
  pivot_data['S1'] = (2 * pivot_data['Pivot Point']) - pivot_data['High']
  pivot_data['R2'] = pivot_data['Pivot Point'] + (pivot_data['High'] - pivot_data['Low'])
  pivot_data['S2'] = pivot_data['Pivot Point'] - (pivot_data['High'] - pivot_data['Low'])
  pivot_data['R3'] = pivot_data['High'] + 2 * (pivot_data['Pivot Point'] - pivot_data['Low'])
  pivot_data['S3'] = pivot_data['Low'] - 2 * (pivot_data['High'] - pivot_data['Pivot Point'])

  # Join the pivot data with the original data frame
  data = pd.concat([data, pivot_data[['Pivot Point', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']]], axis=1)

  # 5-Day Momentum
  data['5-Day Momentum'] = data['Adj Close'] - data['Adj Close'].shift(5)

  # 14-Day Average True Range (ATR)
  data['14-Day ATR'] = data.ta.atr(length=14)

  # 14 Day Simple & Exponential Moving Average
  data['14 Day SMA'] = data['Adj Close'].rolling(window=14).mean()
  data['14 Day EMA'] = data['Adj Close'].ewm(span=14, adjust=False).mean()

  # 14-Day Relative Strength Index (RSI)
  data['14-Day RSI'] = data.ta.rsi(close='Adj Close', length=14)

  # 14 Day Bollinger Bands
  bollinger = data.ta.bbands(length=14, std=2)
  data = data.join(bollinger)

  # On Balance Volume (OBV)
  data['OBV'] = data.ta.obv(close='Adj Close', volume='Volume')

  # 14 Day Fast, Slow & Smoothed Slow Stochastic Indicators
  stoch = data.ta.stoch(high='High', low='Low', close='Adj Close', fastk=14)
  slow_stoch = data.ta.stoch(high='High', low='Low', close='Adj Close', k=3, d=3)
  data = data.join(stoch).join(slow_stoch)

  # Fibonacci Retracement Levels
  data['Fib 38.2%'] = 0.382 * (data['High'].max() - data['Low'].min()) + data['Low'].min()
  data['Fib 50%'] = 0.5 * (data['High'].max() - data['Low'].min()) + data['Low'].min()
  data['Fib 61.8%'] = 0.618 * (data['High'].max() - data['Low'].min()) + data['Low'].min()

  # 3 Day Rate of Change
  data['3 Day ROC'] = data['Adj Close'].pct_change(periods=3)

  # Daily Returns
  data['Daily Returns'] = data['Adj Close'].pct_change()

  # Handling NaN values
  data.dropna(inplace=True)

  return data

def create_dataset(data, target, look_back=1):
  X, Y = [], []
  for i in range(len(data) - look_back - 1):
      X.append(data[i:(i + look_back), :])
      Y.append(target[i + look_back, 0])
  return np.array(X), np.array(Y)



@app.route('/predict', methods=["GET", "POST"])
def predict():
  data = request.get_json()
  start_date = data["startDate"]
  end_date = data["endDate"]
  stock_name = data["stockName"]
  print(f"start date {start_date}, end date {end_date} stock name {stock_name}")
  print(f"start date {type(start_date)}, end date {type(end_date)} stock name {type(stock_name)}")
  yf.pdr_override()
  data = pdr.get_data_yahoo(stock_name, start=start_date, end=end_date)
  last_row = data.iloc[-1]

  open_price = last_row['Open']
  high_price = last_row['High']
  low_price = last_row['Low']
  close_price = last_row['Close']
  adj_close_price = last_row['Adj Close']
  volume = last_row['Volume']

  print("Open:", open_price)
  print("High:", high_price)
  print("Low:", low_price)
  print("Close:", close_price)
  print("Adjusted Close:", adj_close_price)
  print("Volume:", volume)

  # Feature Engineering (Create the Features)
  data = feature_engineering(data)
  print("Data: ", data)

  total_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD_14_21_9', 'MACDh_14_21_9', 'MACDs_14_21_9', 'Daily Returns', '5-Day Momentum', '14-Day ATR', '14 Day SMA', '14 Day EMA', '14-Day RSI', 'BBL_14_2.0', 'BBM_14_2.0', 'BBU_14_2.0', 'BBB_14_2.0', 'BBP_14_2.0', 'OBV', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'STOCHk_3_3_3', 'STOCHd_3_3_3', 'Fib 38.2%', 'Fib 50%', 'Fib 61.8%', '3 Day ROC']

  target = data[['Adj Close']].values
  data = data[total_features].values
  

  # Normalize the data
  scaler_x = MinMaxScaler(feature_range=(0, 1))
  data = scaler_x.fit_transform(data)

  scaler_y = MinMaxScaler(feature_range=(0, 1))
  target = scaler_y.fit_transform(target)

    # Train-test split
  train_size = int(len(data) * 0.8)
  test_size = len(data) - train_size
  train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]
  train_target, test_target = target[0:train_size, :], target[train_size:len(data), :]

  look_back = 10
  X_train, Y_train = create_dataset(train_data, train_target, look_back)
  X_test, Y_test = create_dataset(test_data, test_target, look_back)
  print(f"Testing: {X_test.shape[0]}, {X_test.shape[1]}, {X_test.shape[2]}")

  # Define a dictionary with your custom objects
  custom_objects = {
      'LTCCell': LTCCell  # Ensure your LTCCell class is defined or imported
  }

  if stock_name == 'TSLA':
      
    # Load the model within a custom object scope
    with custom_object_scope(custom_objects):
        model = load_model('tesla_best_model_lnn_2 (1).h5')

    look_back = 10  # Ensure this matches the look_back used during training
    latest_data = X_test[-1].reshape(1, look_back, X_test.shape[2])  # Reshape for the model

    # Make the prediction
    predicted_norm = model.predict(latest_data)
    predicted_price = scaler_y.inverse_transform(predicted_norm)  # Scale back to original price scale

    print(f"Predicted Adjusted Close Price for the next day: ${predicted_price[0][0]:.2f}")
  
  elif stock_name == 'AAPL':
      
    # Load the model within a custom object scope
    with custom_object_scope(custom_objects):
        model = load_model('apple_best_model_lnn_2 (2) (1).h5')

    look_back = 10  # Ensure this matches the look_back used during training
    latest_data = X_test[-1].reshape(1, look_back, X_test.shape[2])  # Reshape for the model

    # Make the prediction
    predicted_norm = model.predict(latest_data)
    predicted_price = scaler_y.inverse_transform(predicted_norm)  # Scale back to original price scale

    print(f"Predicted Adjusted Close Price for the next day: ${predicted_price[0][0]:.2f}")

  return f"{predicted_price[0][0]:.2f}"


if __name__ == "__main__":
  app.run(debug=True)
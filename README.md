
# Stock Market Prediction Using Liquid Neural Networks

This project focuses on predicting the stock prices of Tesla (TSLA) and Apple (AAPL) using advanced machine learning techniques, specifically Liquid Neural Networks (LNN). The goal is to leverage historical stock data to forecast future prices with high accuracy.
![User Interface](https://github.com/HusseinJammal/Liquid-Neural-Networks-in-Stock-Market-Prediction/assets/82166319/a9cb59a3-a9cf-4ec0-bfb7-a0a974b5cdee)

## Project Structure


-   **Data Collection**: Data is sourced from Yahoo Finance using the `yfinance` library, covering daily stock movements from specific start dates up to the present.
-   **Exploratory Data Analysis (EDA)**: Includes candlestick plots with moving averages, correlation heatmaps, box plots, and histograms of daily price changes.
-   **Feature Engineering**: Development of multiple technical indicators to enrich the model's input features.
-   **Model Architecture**: Utilization of Liquid Neural Networks model to predict adjusted close prices.
-   **Evaluation**: Assessment of model performance using metrics like MSE, RMSE, MAE, MAPE, and Directional Accuracy.

## Data


The data for this project is retrieved from Yahoo Finance, focusing on the following stocks:

-   **Apple (AAPL)**: From 2010-01-01 to yesterday.
-   **Tesla (TSLA)**: From 2010-06-29 to yesterday.

Features included are Open, High, Low, Close, Adjusted Close, and Volume. Technical indicators such as MACD, RSI, Bollinger Bands, and others are calculated to enhance the dataset.

## Setup and Installation

To run this project on your local machine, follow these steps:

1.  Clone the repository to your local machine:

    `git clone https://github.com/HusseinJammal/Liquid-Neural-Networks-in-Stock-Market-Prediction.git`

    `cd /Liquid-Neural-Networks-in-Stock-Market-Prediction`

3.  Navigate to the project directory:

    `cd stocks`

4.  Install necessary dependencies:

    `npm i`

5.  Start the application:

    `npm run start`

6.  Run the Python application:

    `npm run app.py`

Ensure you have Python and Node.js installed on your machine, along with necessary libraries such as `yfinance`, `numpy`, `pandas`, `sklearn`, and any specific libraries for LNN or deep learning frameworks you are using.

## Feature Engineering


Detailed feature engineering steps include the creation of:

-   Moving Averages (14 & 21 days for MACD, etc.)
-   Pivot Points, Momentum, and Volatility Indices like ATR and Bollinger Bands
-   Volume indicators like On Balance Volume
-   Oscillators such as Stochastic Indicators
-   Fibonacci Retracement Levels

## Models


### Liquid Neural Networks

Utilization of LNN with configurations for different layers, including dropout and regularization to prevent overfitting.


## Evaluation


The models are evaluated using the following metrics:

-   MSE (Mean Squared Error)
-   RMSE (Root Mean Squared Error)
-   MAE (Mean Absolute Error)
-   MAPE (Mean Absolute Percentage Error)
-   Directional Accuracy

Results from these metrics provide insights into the models' predictive accuracy and performance.

## Contributions


Contributions to this project are welcome. Please fork the repository and submit pull requests with any enhancements or bug fixes.

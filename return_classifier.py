import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    stock = input("Enter stock ticker (e.g., AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    data = load_and_clean_data(stock, start_date, end_date)
    X_train, X_test, y_train, y_test = split_features_and_labels(data, feature_cols = FEATURE_COLUMNS, label_col = 'Label', test_size = 0.2)


def load_and_clean_data(stock, start_date, end_date):
    #download data with auto adjust off
    data = yf.download(stock, start = start_date, end = end_date, auto_adjust = False)

    #flatten a multi-index dataframe if it exists
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    #calculate daily and cumulative returns
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['5 Day Return'] = data['Adj Close'].pct_change(5)
    data['10 Day Return'] = data['Adj Close'].pct_change(10)

    #calculate volatility (rolling 10-day stdev)
    data['Volatility'] = data['Daily Return'].rolling(window = 10).std()

    #calculate 10 and 50 day moving averages + position
    data['10 Day Moving Average'] = data['Adj Close'].rolling(window = 10).mean()
    data['50 Day Moving Average'] = data['Adj Close'].rolling(window = 50).mean()

    data['Signal'] = (data['10 Day Moving Average'] > data['50 Day Moving Average']).astype(int)

    next_return = data['Daily Return'].shift(-1)
    data['Label'] = (next_return > 0).where(next_return.notna())
    data = data.dropna()
    data['Label'] = data['Label'].astype(int)

    return data

#define the feature columns for the split
FEATURE_COLUMNS = ['Daily Return', '5 Day Return', '10 Day Return', 'Volatility', 'Signal']
def split_features_and_labels(data, feature_cols = FEATURE_COLUMNS, label_col = 'Label', test_size = 0.2):
    X = data[feature_cols]
    y = data[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, shuffle = False)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()
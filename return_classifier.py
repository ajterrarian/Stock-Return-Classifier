import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def main():
    stock = input("Enter stock ticker (e.g., AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    data = load_and_clean_data(stock, start_date, end_date)
    X_train, X_test, y_train, y_test = split_features_and_labels(data, feature_cols = FEATURE_COLUMNS, label_col = 'Label', test_size = 0.2)

    majority_position, y_pred_naive, naive_accuracy = naive_baseline(y_train, y_test)

    model, y_pred, y_probability, accuracy = logistic_regression(X_train, X_test, y_train, y_test)


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

    data = data.replace([np.inf, -np.inf], np.nan)
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

def naive_baseline(y_train, y_test):
    #find majority position and flip all values to match
    majority_position = y_train.value_counts().idxmax()
    y_pred_naive = pd.Series(majority_position, index = y_test.index, name = 'Naive Position')

    #test accuracy
    naive_accuracy = accuracy_score(y_test, y_pred_naive)
    print(f"Accuracy score: {naive_accuracy:.4f}\n")

    report_dict = classification_report(y_test, y_pred_naive, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.iloc[:, :3] = report_df.iloc[:, :3].replace(0.0, np.nan) # Replace precision, recall, f1
    print(report_df.to_string())

    print("Baseline Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_naive))

    return majority_position, y_pred_naive, naive_accuracy

def logistic_regression(X_train, X_test, y_train, y_test):
    #scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #start and train the model
    model = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
    model.fit(X_train_scaled, y_train)

    #make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    y_probability = model.predict_proba(X_test_scaled)

    #evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {accuracy:.4f}\n")
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.iloc[:, :3] = report_df.iloc[:, :3].replace(0.0, np.nan) # Replace precision, recall, f1
    print(report_df.to_string())
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, y_pred, y_probability, accuracy


if __name__ == "__main__":
    main()
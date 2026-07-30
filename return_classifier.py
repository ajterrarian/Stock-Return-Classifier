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

    rf_model, rf_y_pred, rf_y_probability, rf_accuracy, rf_importances = random_forest(X_train, X_test, y_train, y_test)

def fetch_price_data(stock, start_date, end_date):
    #network I/O -- not unit tested directly, only exercised by a real run
    data = yf.download(stock, start = start_date, end = end_date, auto_adjust = False)

    #flatten a multi-index dataframe if it exists
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data['Adj Close']   #only the column compute_features actually needs

def compute_features(prices):
    #pure computation: Series of prices in, DataFrame of engineered features + Label out.
    #NaNs (warm-up at the start, missing "tomorrow" at the end) are left in place on purpose --
    #this is what makes it testable on a small synthetic Series.
    data = pd.DataFrame(index=prices.index)
    data['Adj Close'] = prices

    #calculate daily and cumulative returns
    data['Daily Return'] = prices.pct_change()
    data['5 Day Return'] = prices.pct_change(5)
    data['10 Day Return'] = prices.pct_change(10)

    #calculate volatility (rolling 10-day stdev)
    data['Volatility'] = data['Daily Return'].rolling(window = 10).std()

    #calculate 10 and 50 day moving averages + position
    data['10 Day Moving Average'] = prices.rolling(window = 10).mean()
    data['50 Day Moving Average'] = prices.rolling(window = 50).mean()

    data['Signal'] = (data['10 Day Moving Average'] > data['50 Day Moving Average']).astype(int)

    #label: 1 if tomorrow's return is positive, else 0 -- .where() preserves NaN
    #for the last row instead of letting ">" silently turn it into False
    next_return = data['Daily Return'].shift(-1)
    data['Label'] = (next_return > 0).where(next_return.notna())

    return data

def finalize_features(data):
    #cleanup: inf -> NaN so dropna() actually catches it, drop remaining NaNs,
    #then cast Label to int now that nothing NaN is left to collide with an int dtype
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    data['Label'] = data['Label'].astype(int)
    return data

def load_and_clean_data(stock, start_date, end_date):
    #orchestrator: fetch -> compute -> finalize
    prices = fetch_price_data(stock, start_date, end_date)
    data = compute_features(prices)
    data = finalize_features(data)
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
    print("Baseline Classification Report:")
    print(classification_report(y_test, y_pred_naive, zero_division=0))
    print("Baseline Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_naive))

    return majority_position, y_pred_naive, naive_accuracy

def logistic_regression(X_train, X_test, y_train, y_test):
    #scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #start and train the model
    model = LogisticRegression(solver = 'lbfgs', max_iter = 1000, random_state = 42)
    model.fit(X_train_scaled, y_train)

    #make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    y_probability = model.predict_proba(X_test_scaled)

    #evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, y_pred, y_probability, accuracy

def random_forest(X_train, X_test, y_train, y_test, max_depth = 5):
    model = RandomForestClassifier(n_estimators = 100, max_depth = max_depth, random_state = 42)
    model.fit(X_train, y_train)

    #make predictions on the test set
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)

    #evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    #feature importance
    importances = pd.Series(model.feature_importances_, index = FEATURE_COLUMNS).sort_values(ascending=False)
    print("Feature Importances:")
    print(importances)

    return model, y_pred, y_probability, accuracy, importances

if __name__ == "__main__":
    main()
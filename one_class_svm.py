import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from main import get_results_store  # Import function to retrieve stored results


def detect_outliers(df, feature_columns, nu=0.05, kernel='rbf'):
    """
    Detect outliers using One-Class SVM on each ticker's features.

    Args:
        df (pd.DataFrame): Multi-index DataFrame with financial features.
        feature_columns (list): Features used for outlier detection.
        nu (float): An upper bound on the fraction of training errors (outliers).
        kernel (str): Kernel type to be used in the algorithm.

    Returns:
        dict: A dictionary with ticker as key and outlier ratio as value.
    """
    tickers = df.columns.levels[0]
    outlier_results = {}

    for ticker in tickers:
        # Ensure required features exist
        available_features = [col for col in feature_columns if col in df[ticker].columns]
        if not available_features:
            print(f"Warning: No specified feature columns found for {ticker}.")
            continue

        features = df[ticker][available_features].dropna()
        if features.empty:
            print(f"Warning: Feature data is empty for {ticker}.")
            continue

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Fit One-Class SVM
        model = OneClassSVM(nu=nu, kernel=kernel)
        preds = model.fit_predict(scaled_features)

        # -1 → outlier, +1 → inlier
        outlier_ratio = (preds == -1).sum() / len(preds)
        outlier_results[(ticker, 'outlier_ratio')] = outlier_ratio
        print(f"{ticker}: Outlier ratio = {outlier_ratio:.4f}")

    return outlier_results


def run(identifier, feature_columns, nu=0.05, kernel='rbf'):
    """
    Load stored data and run One-Class SVM outlier detection.

    Args:
        identifier (str): Unique key for stored data.
        feature_columns (list): List of features to use.
        nu (float): Upper bound on the fraction of outliers.
        kernel (str): Kernel type ('rbf', 'linear', 'poly', etc.)

    Returns:
        dict or None: Outlier detection results, or None if invalid.
    """
    results_store = get_results_store()
    stored_key = f"{identifier}"

    if stored_key not in results_store:
        print(f"Error: No data found for identifier '{stored_key}' in results_store.")
        return None

    df = results_store[stored_key]

    if df is None or df.empty:
        print(f"Error: DataFrame '{stored_key}' is empty.")
        return None

    return detect_outliers(df, feature_columns, nu=nu, kernel=kernel)

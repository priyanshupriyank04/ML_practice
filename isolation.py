import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from main import get_results_store  # Import function to retrieve stored results


def perform_isolation_forest(df, feature_columns, contamination=0.05):
    """
    Perform anomaly detection using Isolation Forest on the given multi-level DataFrame.

    Args:
        df (pd.DataFrame): Multi-level DataFrame containing financial metrics.
        feature_columns (list): List of column names to be used as features.
        contamination (float): The proportion of outliers in the data (default is 0.05).

    Returns:
        dict: A dictionary with ticker as key and a DataFrame with anomaly score and outlier flag.
    """
    tickers = df.columns.levels[0]  # Extract tickers from level 0
    results = {}

    for ticker in tickers:
        # Check if feature columns are available for the ticker
        available_features = [col for col in feature_columns if col in df[ticker].columns]
        if not available_features:
            print(f"Warning: No specified feature columns found for {ticker}.")
            continue

        # Extract and scale features
        features = df[ticker][available_features]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_scores = iso_forest.decision_function(scaled_features)
        predictions = iso_forest.predict(scaled_features)

        # -1 means outlier, 1 means inlier
        results[ticker] = pd.DataFrame({
            'anomaly_score': anomaly_scores,
            'is_outlier': (predictions == -1).astype(int)
        }, index=df[ticker].index)

        print(f"{ticker}: Detected {results[ticker]['is_outlier'].sum()} outliers.")

    return results


def run(identifier, feature_columns, contamination=0.05):
    """
    Retrieve stored data and apply Isolation Forest for anomaly detection.

    Args:
        identifier (str): Unique identifier to fetch the corresponding DataFrame.
        feature_columns (list): List of feature columns to use.
        contamination (float, optional): Proportion of outliers. Defaults to 0.05.

    Returns:
        dict or None: Dictionary of results per ticker if data exists, else None.
    """
    results_store = get_results_store()
    stored_key = f"{identifier}"

    if stored_key not in results_store:
        print(f"Error: No data found for identifier '{stored_key}' in results_store.")
        return None

    df = results_store[stored_key]

    if df is None or df.empty:
        print(f"Error: DataFrame '{stored_key}' is empty. Check CSV data and columns.")
        return None

    return perform_isolation_forest(df, feature_columns, contamination)

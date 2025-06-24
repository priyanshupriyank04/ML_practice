import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from main import get_results_store  # Import function to retrieve stored results


def perform_lof(df, feature_columns, n_neighbors=20):
    """
    Perform Local Outlier Factor detection on the given multi-level DataFrame.

    Args:
        df (pd.DataFrame): Multi-level DataFrame containing financial metrics.
        feature_columns (list): List of column names to be used as features.
        n_neighbors (int): Number of neighbors to use for LOF. Default is 20.

    Returns:
        dict: Dictionary with outlier counts per ticker.
    """
    tickers = df.columns.levels[0]  # Extract tickers from level 0
    outlier_summary = {}

    for ticker in tickers:
        # Check if feature columns are available for the ticker
        available_features = [col for col in feature_columns if col in df[ticker].columns]
        if not available_features:
            print(f"Warning: No specified feature columns found for {ticker}.")
            continue

        features = df[ticker][available_features].dropna()
        if features.empty:
            print(f"Warning: No valid feature data for {ticker}.")
            continue

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Apply LOF
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        flags = lof.fit_predict(scaled_features)  # -1 = outlier, 1 = inlier

        # Store results in the DataFrame (optional: persist back if needed)
        result_df = features.copy()
        result_df["lof_flag"] = flags
        outliers = (flags == -1).sum()
        outlier_summary[ticker] = {"outliers": int(outliers), "total": len(flags)}

        print(f"{ticker}: {outliers} outliers detected out of {len(flags)}")

    return outlier_summary


def run(identifier, feature_columns, n_neighbors=20):
    """
    Retrieve stored data and perform LOF outlier detection.

    Args:
        identifier (str): Unique identifier to fetch the corresponding DataFrame.
        feature_columns (list): List of feature columns to use for outlier detection.
        n_neighbors (int, optional): Number of neighbors to use for LOF. Defaults to 20.

    Returns:
        dict or None: Dictionary summarizing outliers per ticker, or None if data is missing.
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

    return perform_lof(df, feature_columns, n_neighbors)

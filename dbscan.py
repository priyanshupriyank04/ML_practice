import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from main import get_results_store  # Import function to retrieve stored results


def perform_dbscan(df, feature_columns, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on the given multi-level DataFrame and 
    identify anomalies (label = -1).

    Args:
        df (pd.DataFrame): Multi-level DataFrame containing financial metrics.
        feature_columns (list): List of column names to be used as features.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.

    Returns:
        dict: A dictionary with ticker as key and a summary containing number of clusters and anomalies.
    """
    tickers = df.columns.levels[0]
    dbscan_summary = {}

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

        # Optional: Reduce dimensions with PCA (useful if many features)
        pca = PCA(n_components=1)
        reduced_features = pca.fit_transform(scaled_features)

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(reduced_features)

        # Count anomalies (label = -1) and clusters
        n_outliers = sum(label == -1 for label in cluster_labels)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        # Store label in original DataFrame (optional)
        df[ticker]['dbscan_label'] = cluster_labels

        print(f"{ticker} - clusters={n_clusters}, outliers={n_outliers}")
        dbscan_summary[ticker] = {
            "n_clusters": n_clusters,
            "n_outliers": n_outliers
        }

    return dbscan_summary


def run(identifier, feature_columns, eps=0.5, min_samples=5):
    """
    Retrieve stored data and perform DBSCAN clustering on specified feature columns.

    Args:
        identifier (str): Unique identifier to fetch the corresponding DataFrame.
        feature_columns (list): List of feature columns to use for clustering.
        eps (float, optional): DBSCAN epsilon parameter.
        min_samples (int, optional): DBSCAN minimum samples parameter.

    Returns:
        dict or None: Dictionary of cluster/outlier stats if data exists, else None.
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

    return perform_dbscan(df, feature_columns, eps, min_samples)

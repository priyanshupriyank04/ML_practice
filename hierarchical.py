import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from main import get_results_store  # Import function to retrieve stored results


def perform_hierarchical(df, feature_columns, n_clusters=3):
    """
    Perform Agglomerative Hierarchical clustering on the given multi-level DataFrame.

    Args:
        df (pd.DataFrame): Multi-level DataFrame containing financial metrics.
        feature_columns (list): List of column names to be used as features.
        n_clusters (int): The number of clusters to find.

    Returns:
        dict: A dictionary with ticker as key and cluster count summary.
    """
    tickers = df.columns.levels[0]
    cluster_summary = {}

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

        # Optional: Reduce dimensions with PCA
        pca = PCA(n_components=1)
        reduced_features = pca.fit_transform(scaled_features)

        # Perform Agglomerative Clustering
        model = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = model.fit_predict(reduced_features)

        # Store label in original DataFrame (optional)
        df[ticker]['hierarchical_label'] = cluster_labels

        # Count distribution of labels
        label_counts = pd.Series(cluster_labels).value_counts().to_dict()
        print(f"{ticker} - cluster distribution: {label_counts}")
        cluster_summary[ticker] = label_counts

    return cluster_summary


def run(identifier, feature_columns, n_clusters=3):
    """
    Retrieve stored data and perform hierarchical clustering.

    Args:
        identifier (str): Unique identifier to fetch the corresponding DataFrame.
        feature_columns (list): List of feature columns to use for clustering.
        n_clusters (int, optional): Number of clusters. Defaults to 3.

    Returns:
        dict or None: Dictionary of cluster label counts if data exists, else None.
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

    return perform_hierarchical(df, feature_columns, n_clusters)

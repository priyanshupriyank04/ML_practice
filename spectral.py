import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from main import get_results_store  # Function to retrieve stored results


def perform_clustering(df, feature_columns, range_n_clusters):
    """
    Perform Spectral Clustering on the given multi-level DataFrame and evaluate clusters using silhouette score.

    Args:
        df (pd.DataFrame): Multi-level DataFrame containing financial metrics.
        feature_columns (list): List of column names to be used as features.
        range_n_clusters (tuple): Range of cluster numbers to evaluate (start, end).

    Returns:
        dict: A dictionary with cluster sizes as keys and silhouette scores as values.
    """
    tickers = df.columns.levels[0]  # Extract tickers from level 0
    silhouette_scores = {}

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

        # Reduce dimensions using PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(scaled_features)

        # Perform spectral clustering for different cluster sizes
        for n_clusters in range(range_n_clusters[0], range_n_clusters[1] + 1):
            try:
                model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
                cluster_labels = model.fit_predict(reduced_features)
                silhouette_avg = silhouette_score(reduced_features, cluster_labels)

                # Store silhouette score
                silhouette_scores[(ticker, f"clust_spectral_{n_clusters}")] = silhouette_avg
                print(f"{ticker} - Spectral n_clusters={n_clusters}: {silhouette_avg:.4f}")
            except Exception as e:
                print(f"Error clustering {ticker} with n_clusters={n_clusters}: {e}")

    return silhouette_scores


def run(identifier, feature_columns, range_n_clusters=(2, 10)):
    """
    Retrieve stored data and perform Spectral Clustering on specified feature columns.

    Args:
        identifier (str): Unique identifier to fetch the corresponding DataFrame.
        feature_columns (list): List of feature columns to use for clustering.
        range_n_clusters (tuple, optional): Range of cluster sizes to evaluate. Defaults to (2, 10).

    Returns:
        dict or None: Dictionary of silhouette scores if data exists, else None.
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

    return perform_clustering(df, feature_columns, range_n_clusters)

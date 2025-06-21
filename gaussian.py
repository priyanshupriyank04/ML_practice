import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from main import get_results_store  # Import function to retrieve stored results


def perform_gmm(df, feature_columns, n_components=3):
    """
    Perform Gaussian Mixture Model clustering on the given multi-level DataFrame.

    Args:
        df (pd.DataFrame): Multi-level DataFrame containing financial metrics.
        feature_columns (list): List of column names to be used as features.
        n_components (int): Number of mixture components (clusters).

    Returns:
        dict: A dictionary with ticker as key and cluster label distribution.
    """
    tickers = df.columns.levels[0]
    gmm_summary = {}

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

        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        cluster_labels = gmm.fit_predict(reduced_features)

        # Store cluster labels (optional)
        df[ticker]['gmm_label'] = cluster_labels

        # Optional: Store cluster probabilities if needed
        # probs = gmm.predict_proba(reduced_features)
        # for i in range(n_components):
        #     df[ticker][f'gmm_prob_{i}'] = probs[:, i]

        # Count cluster distribution
        label_counts = pd.Series(cluster_labels).value_counts().to_dict()
        print(f"{ticker} - GMM cluster distribution: {label_counts}")
        gmm_summary[ticker] = label_counts

    return gmm_summary


def run(identifier, feature_columns, n_components=3):
    """
    Retrieve stored data and perform GMM clustering.

    Args:
        identifier (str): Unique identifier to fetch the corresponding DataFrame.
        feature_columns (list): List of feature columns to use for clustering.
        n_components (int, optional): Number of clusters (Gaussian components).

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

    return perform_gmm(df, feature_columns, n_components)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from main import get_results_store  # Import function to retrieve stored results


def perform_pca(df, feature_columns, n_components=2):
    """
    Perform PCA on the given multi-level DataFrame for each ticker.

    Args:
        df (pd.DataFrame): Multi-level DataFrame with financial metrics.
        feature_columns (list): List of column names to be used as features.
        n_components (int): Number of principal components to extract.

    Returns:
        dict: A dictionary containing PCA results per ticker.
    """
    tickers = df.columns.levels[0]
    pca_results = {}

    for ticker in tickers:
        # Check if specified features exist
        available_features = [col for col in feature_columns if col in df[ticker].columns]
        if not available_features:
            print(f"Warning: No specified feature columns found for {ticker}.")
            continue

        features = df[ticker][available_features]

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Perform PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_features)

        # Store results
        pca_results[ticker] = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'components': pca.components_.tolist(),
            'transformed_data': principal_components.tolist()
        }

        print(f"{ticker}: Explained variance ratio = {pca.explained_variance_ratio_}")

    return pca_results


def run(identifier, feature_columns, n_components=2):
    """
    Load stored data and perform PCA transformation.

    Args:
        identifier (str): Unique key to retrieve the dataframe.
        feature_columns (list): List of column names to be used.
        n_components (int, optional): Number of components to retain. Defaults to 2.

    Returns:
        dict or None: Dictionary of PCA results if successful, else None.
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

    return perform_pca(df, feature_columns, n_components)

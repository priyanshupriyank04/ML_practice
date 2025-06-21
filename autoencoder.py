import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from main import get_results_store


def build_autoencoder(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def perform_autoencoder(df, feature_columns, epochs=50, batch_size=32, threshold=0.01):
    tickers = df.columns.levels[0]
    outlier_summary = {}

    for ticker in tickers:
        available_features = [col for col in feature_columns if col in df[ticker].columns]
        if not available_features:
            print(f"Warning: No specified feature columns found for {ticker}.")
            continue

        features = df[ticker][available_features].dropna()
        if features.empty:
            print(f"Warning: No valid feature data for {ticker}.")
            continue

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)

        model = build_autoencoder(input_dim=scaled_features.shape[1])
        model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0)

        reconstructed = model.predict(scaled_features)
        mse = np.mean(np.square(scaled_features - reconstructed), axis=1)

        anomaly_flags = mse > threshold
        outliers = np.sum(anomaly_flags)

        print(f"{ticker}: {outliers} anomalies (threshold={threshold})")

        outlier_summary[ticker] = {
            "outliers": int(outliers),
            "total": len(mse),
            "mean_reconstruction_error": float(np.mean(mse))
        }

        K.clear_session()  # Free up memory

    return outlier_summary


def run(identifier, feature_columns, epochs=50, batch_size=32, threshold=0.01):
    results_store = get_results_store()
    stored_key = f"{identifier}"

    if stored_key not in results_store:
        print(f"Error: No data found for identifier '{stored_key}' in results_store.")
        return None

    df = results_store[stored_key]
    if df is None or df.empty:
        print(f"Error: DataFrame '{stored_key}' is empty.")
        return None

    return perform_autoencoder(df, feature_columns, epochs, batch_size, threshold)

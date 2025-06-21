import pandas as pd
from main import get_results_store  # Import function instead of variable


def calculate_linear_regression(df, column, window):
    """
    Calculate the Linear Regression trendline for all tickers in a multi-index DataFrame.

    Args:
        df (pd.DataFrame): The input multi-index DataFrame with tickers as level 0.
        column (str): The column name to apply Linear Regression on (e.g., 'close').
        window (int): The rolling window size.

    Returns:
        pd.DataFrame: The DataFrame with additional Linear Regression trendline columns.
    """
    tickers = df.columns.levels[0]  # Extract tickers from level 0 of the MultiIndex columns

    def lin_reg(series):
        """
        Compute the predicted value at the last point of a rolling window 
        using a simple linear regression (least squares method).

        Args:
            series (pd.Series): A rolling window series of values.

        Returns:
            float or None: The predicted value at the last point of the window, or None if insufficient data.
        """
        if len(series) < window:
            return None

        x = pd.Series(range(len(series)))  # Time steps (0, 1, 2, ...)
        y = series.values
        x_mean, y_mean = x.mean(), y.mean()

        # Compute slope (b1) and intercept (b0) using least squares regression
        b1 = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        b0 = y_mean - b1 * x_mean

        return b0 + b1 * (window - 1)  # Predicted value at the last point

    for ticker in tickers:
        if column in df[ticker].columns:  # Check if the specified column exists
            df[(ticker, f"{column}_lin_reg")] = df[(ticker, column)].rolling(window=window).apply(
                lin_reg, raw=True
            )
        else:
            print(f"Warning: '{column}' column not found for {ticker}.")

    return df


def run(identifier, column="close", window=14):
    """
    Retrieve stored data and compute the Linear Regression trendline for the specified column.

    Args:
        identifier (str): Unique identifier to fetch the corresponding DataFrame.
        column (str, optional): Column name to apply Linear Regression on. Defaults to "close".
        window (int, optional): Window size for Linear Regression calculation. Defaults to 14.

    Returns:
        pd.DataFrame or None: DataFrame with Linear Regression trendline added if data exists, else None.
    """
    results_store = get_results_store()  # Retrieve stored results
    stored_key = f"{identifier}"  # Ensure correct key format

    if stored_key not in results_store:
        print(f"Error: No data found for identifier '{stored_key}' in results_store.")
        return None

    df = results_store[stored_key]

    if df is None or df.empty:
        print(f"Error: DataFrame '{stored_key}' is empty. Check CSV data and columns.")
        return None

    return calculate_linear_regression(df, column, window)

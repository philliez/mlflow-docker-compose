import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- Configuration ---
OUTPUT_X_FILE = 'X_data.npz'
OUTPUT_Y_FILE = 'y_data.npz'

# Base columns from 1_consolidate_and_process.py
BASE_COLS = ['open', 'high', 'low', 'close', 'volume', 'histogram', 'signal', 'macd']

def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(df, window=20, std_dev=2):
    df['SMA_20_close'] = df['close'].rolling(window=window).mean()
    df['upper_band'] = df['SMA_20_close'] + (df['close'].rolling(window=window).std() * std_dev)
    df['lower_band'] = df['SMA_20_close'] - (df['close'].rolling(window=window).std() * std_dev)
    return df

def calculate_stochastic_oscillator(df, window=14):
    low_min = df['low'].rolling(window=window).min()
    high_max = df['high'].rolling(window=window).max()
    df['stochastic_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()
    return df

def calculate_atr(df, window=14):
    df['high-low'] = df['high'] - df['low']
    df['high-close_prev'] = np.abs(df['high'] - df['close'].shift(1))
    df['low-close_prev'] = np.abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high-low', 'high-close_prev', 'low-close_prev']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=window).mean()
    return df

def apply_feature_engineering(list_of_dfs, list_of_labels):
    engineered_features_list = []
    final_labels = []

    for i, df in tqdm(enumerate(list_of_dfs), total=len(list_of_dfs), desc="Applying Feature Engineering"):
        # Ensure the DataFrame has the base columns
        if not all(col in df.columns for col in BASE_COLS):
            print(f"Skipping DataFrame {i} due to missing base columns.")
            continue

        # --- Feature Engineering Examples ---
        # 1. Simple Moving Average (SMA) for 'close' price
        df['SMA_5_close'] = df['close'].rolling(window=5).mean()
        df['SMA_10_close'] = df['close'].rolling(window=10).mean()

        # 2. Candle Body Size
        df['candle_body_size'] = abs(df['open'] - df['close'])

        # 3. Relative Strength Index (RSI)
        df['RSI_14'] = calculate_rsi(df)

        # 4. Lagged features for 'close' price
        df['close_lag_1'] = df['close'].shift(1)
        df['close_lag_2'] = df['close'].shift(2)
        df['close_lag_3'] = df['close'].shift(3)
        df['close_lag_4'] = df['close'].shift(4)
        df['close_lag_5'] = df['close'].shift(5)

        # 5. Bollinger Bands
        df = calculate_bollinger_bands(df)

        # 6. Stochastic Oscillator
        df = calculate_stochastic_oscillator(df)

        # 7. Average True Range (ATR)
        df = calculate_atr(df)

        # --- Drop rows with NaN values created by feature engineering ---
        # This is crucial as many indicators will have NaN for the first few rows
        df = df.dropna()

        # If, after dropping NaNs, the DataFrame is empty or too small, skip it
        if df.empty or len(df) < 10: # Example threshold, adjust as needed
            continue

        # Flatten the DataFrame into a single row for XGBoost
        flattened_features = df.values.flatten()
        engineered_features_list.append(flattened_features)
        final_labels.append(list_of_labels[i])

    if not engineered_features_list:
        print("Error: No data remaining after feature engineering and NaN removal.")
        return None, None, None

    X_data_np = np.array(engineered_features_list)
    y_data_np = np.array(final_labels)

    # Determine the final feature columns after engineering and flattening
    # This will be a long list of feature_name_timestep
    # Get columns from the last processed DataFrame (assuming all have same columns after FE)
    if list_of_dfs:
        sample_df = list_of_dfs[-1] # Use the last processed df as a template
        # Re-apply feature engineering to a dummy df to get final columns after dropna
        temp_df = sample_df.copy()
        temp_df['SMA_5_close'] = temp_df['close'].rolling(window=5).mean()
        temp_df['SMA_10_close'] = temp_df['close'].rolling(window=10).mean()
        temp_df['candle_body_size'] = abs(temp_df['open'] - temp_df['close'])
        temp_df['RSI_14'] = calculate_rsi(temp_df)
        temp_df['close_lag_1'] = temp_df['close'].shift(1)
        temp_df['close_lag_2'] = temp_df['close'].shift(2)
        temp_df['close_lag_3'] = temp_df['close'].shift(3)
        temp_df['close_lag_4'] = temp_df['close'].shift(4)
        temp_df['close_lag_5'] = temp_df['close'].shift(5)
        temp_df = calculate_bollinger_bands(temp_df)
        temp_df = calculate_stochastic_oscillator(temp_df)
        temp_df = calculate_atr(temp_df)
        temp_df = temp_df.dropna()
        
        original_cols_after_fe = temp_df.columns.tolist()
        num_timesteps_after_fe = len(temp_df) # Number of timesteps remaining after dropna

        final_feature_cols = []
        for col in original_cols_after_fe:
            for i in range(num_timesteps_after_fe):
                final_feature_cols.append(f"{col}_t{i}")
    else:
        final_feature_cols = [] # Should not happen if engineered_features_list is not empty

    print(f"\nProcessed {len(X_data_np)} valid samples after feature engineering.")
    print(f"Final feature columns count: {len(final_feature_cols)}")
    print(f"Data shape (X): {X_data_np.shape}")
    print(f"Labels shape (y): {y_data_np.shape}")

    np.savez_compressed(OUTPUT_X_FILE, data=X_data_np, feature_names=final_feature_cols)
    np.savez_compressed(OUTPUT_Y_FILE, data=y_data_np)
    print(f"\nSuccessfully saved engineered data to {OUTPUT_X_FILE} and {OUTPUT_Y_FILE}")

    return X_data_np, y_data_np, final_feature_cols

if __name__ == '__main__':
    # This script is intended to be called by an orchestrator, but for testing:
    print("This script is designed to be imported and called. Run the orchestrator.")

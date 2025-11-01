import numpy as np
import pandas as pd
from tradestorun.feature_engineer import apply_feature_engineering

def main():
    # Load the processed data
    try:
        with np.load('X_data.npz', allow_pickle=True) as data:
            X_data_list = data['arr_0']
        with np.load('y_data.npz', allow_pickle=True) as data:
            y_data = data['arr_0']
    except FileNotFoundError:
        print("Error: X_data.npz or y_data.npz not found. Please run the consolidation and processing script first.")
        return

    # Convert numpy arrays back to list of dataframes
    list_of_dfs = [pd.DataFrame(arr, columns=['open', 'high', 'low', 'close', 'volume', 'histogram', 'signal', 'macd']) for arr in X_data_list]

    # Apply feature engineering
    X_data_np, y_data_np, final_feature_cols = apply_feature_engineering(list_of_dfs, y_data)

    # Save the final data
    if X_data_np is not None and y_data_np is not None:
        np.savez_compressed('X_data_engineered.npz', data=X_data_np, feature_names=final_feature_cols)
        np.savez_compressed('y_data_engineered.npz', data=y_data_np)
        print("\nSuccessfully saved final engineered data to X_data_engineered.npz and y_data_engineered.npz")

if __name__ == '__main__':
    main()


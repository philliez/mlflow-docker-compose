import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Configuration ---
def process_data(base_data_dir, expected_rows, expected_cols):
    all_file_paths = []
    for root, _, files in os.walk(base_data_dir):
        for file in files:
            if file.endswith('.csv'):
                all_file_paths.append(os.path.join(root, file))

    if not all_file_paths:
        print(f"Error: No CSV files found in directory tree: '{base_data_dir}'.")
        return [], []

    print(f"Found {len(all_file_paths)} total CSV files to process.")
    all_file_paths.sort() # Sort chronologically for the train/val split

    X_data_dfs = [] # List to store processed DataFrames
    y_data = []

    for file_path in tqdm(all_file_paths, desc="Processing files"):
        try:
            filename = os.path.basename(file_path)
            if 'win' in filename.lower():
                label = 1
            elif 'loss' in filename.lower():
                label = 0
            else:
                continue

            df = pd.read_csv(file_path, index_col=0)

            if len(df) != expected_rows:
                continue

            df = df.iloc[::-1]

            features_df = df[expected_cols]
            
            X_data_dfs.append(features_df)
            y_data.append(label)

        except Exception as e:
            print(f"Skipping file {file_path} due to error: {e}")

    if not X_data_dfs:
        print("Error: No valid data was processed.")
        return [], []

    print(f"\nProcessed {len(X_data_dfs)} valid samples.")
    print(f"Feature columns being used: {expected_cols}")

    return X_data_dfs, y_data

if __name__ == '__main__':
    # Define EXPECTED_ROWS and EXPECTED_COLS for direct execution
    EXPECTED_ROWS = 480
    EXPECTED_COLS = ['open', 'high', 'low', 'close', 'volume', 'histogram', 'signal', 'macd']

    all_processed_dfs = []
    all_labels = []

    for month in ['06', '07', '08']:
        for day in range(1, 32):
            date_str = f'2025-{month}-{day:02d}'
            base_dir = os.path.join(os.path.dirname(__file__), date_str)
            if os.path.exists(base_dir):
                print(f'Processing data for {date_str}')
                processed_dfs, labels = process_data(base_dir, EXPECTED_ROWS, EXPECTED_COLS)
                all_processed_dfs.extend(processed_dfs)
                all_labels.extend(labels)

    print(f"Returned {len(all_processed_dfs)} dataframes and {len(all_labels)} labels.")

    # Save the processed data
    if all_processed_dfs:
        # Assuming the dataframes have the same columns, we can concatenate them.
        # However, the output of process_data is a list of dataframes, not a single one.
        # We need to handle this properly. For now, let's save them as a list of numpy arrays.
        X_data = np.array([df.to_numpy() for df in all_processed_dfs])
        y_data = np.array(all_labels)

        np.savez_compressed('X_data.npz', X_data)
        np.savez_compressed('y_data.npz', y_data)

        print("Saved processed data to X_data.npz and y_data.npz")

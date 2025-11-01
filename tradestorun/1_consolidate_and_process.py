import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = os.path.expanduser('~/serpai/tradestorun/Session 1') # Adjusted to your structure
OUTPUT_X_FILE = 'X_data.npz'
OUTPUT_Y_FILE = 'y_data.npz'
EXPECTED_ROWS = 480
# --- ALIGNMENT: These are the exact columns your script saves ---
# Note: 'timestamp' becomes the index after loading, so we don't list it here.
EXPECTED_COLS = ['open', 'high', 'low', 'close', 'volume', 'histogram', 'signal', 'macd']

def process_data():
    if not os.path.isdir(DATA_DIR):
        print(f"Error: The specified data directory does not exist: {DATA_DIR}")
        return

    all_file_paths = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.csv'):
                all_file_paths.append(os.path.join(root, file))

    if not all_file_paths:
        print(f"Error: No CSV files found in directory tree: '{DATA_DIR}'.")
        return

    print(f"Found {len(all_file_paths)} total CSV files to process.")
    all_file_paths.sort() # Sort chronologically for the train/val split

    X_data = []
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

            # Load the CSV, setting the first column as the timestamp index
            df = pd.read_csv(file_path, index_col=0)

            if len(df) != EXPECTED_ROWS:
                continue

            # --- CRITICAL ALIGNMENT STEP ---
            # Your script saves the data reversed. We must un-reverse it here
            # so the model sees it in correct chronological order (oldest to newest).
            df = df.iloc[::-1]

            # Select only the features the model will be trained on.
            features_np = df[EXPECTED_COLS].values
            
            X_data.append(features_np)
            y_data.append(label)

        except Exception as e:
            print(f"Skipping file {file_path} due to error: {e}")

    if not X_data:
        print("Error: No valid data was processed. Output files will not be created.")
        return

    X_data_np = np.array(X_data)
    y_data_np = np.array(y_data)

    print(f"\nProcessed {len(X_data_np)} valid samples.")
    print(f"Feature columns being used: {EXPECTED_COLS}")
    print(f"Data shape (X): {X_data_np.shape}")
    print(f"Labels shape (y): {y_data_np.shape}")

    np.savez_compressed(OUTPUT_X_FILE, data=X_data_np)
    np.savez_compressed(OUTPUT_Y_FILE, data=y_data_np)
    print(f"\nSuccessfully saved data to {OUTPUT_X_FILE} and {OUTPUT_Y_FILE}")

if __name__ == '__main__':
    process_data()

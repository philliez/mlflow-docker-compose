import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import zipfile
import io

# --- Configuration ---
ZIP_FILE = 'ohlc_data.zip'
OUTPUT_X_FILE = 'X_data.npz'
OUTPUT_Y_FILE = 'y_data.npz'
EXPECTED_ROWS = 480
EXPECTED_COLS = ['open', 'high', 'low', 'close', 'volume', 'histogram', 'signal', 'macd']

def process_data():
    if not os.path.exists(ZIP_FILE):
        print(f"Error: The specified zip file does not exist: {ZIP_FILE}")
        return

    X_data = []
    y_data = []

    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        all_file_paths = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        print(f"Found {len(all_file_paths)} total CSV files to process in the zip file.")
        all_file_paths.sort() # Sort chronologically

        for file_path in tqdm(all_file_paths, desc="Processing files"):
            try:
                filename = os.path.basename(file_path)
                if 'win' in filename.lower():
                    label = 1
                elif 'loss' in filename.lower():
                    label = 0
                else:
                    continue

                with zip_ref.open(file_path) as csv_file:
                    # Use io.TextIOWrapper to decode the file content as text
                    csv_content = io.TextIOWrapper(csv_file, 'utf-8')
                    df = pd.read_csv(csv_content, index_col=0)

                if len(df) != EXPECTED_ROWS:
                    continue

                df = df.iloc[::-1]
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

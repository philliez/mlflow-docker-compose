import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = 'btcusdt_win_loss_model.keras'
SCALER_PATH = 'scaler.joblib'
EXPECTED_ROWS = 480
EXPECTED_COLS = ['open', 'high', 'low', 'close', 'volume', 'histogram', 'signal', 'macd']

def process_single_file(file_path, model, scaler):
    """
    Processes a single raw CSV file and returns a prediction score and any errors.
    """
    try:
        df = pd.read_csv(file_path, index_col=0)
        if len(df) != EXPECTED_ROWS:
            return None, f"Incorrect row count ({len(df)})"
        
        df = df.iloc[::-1]  # Reverse to chronological order
        features = df[EXPECTED_COLS]
        scaled_features = scaler.transform(features.values)
        reshaped_data = np.expand_dims(scaled_features, axis=0)
        prediction = model.predict(reshaped_data, verbose=0)[0][0]
        return prediction, None
    except Exception as e:
        return None, str(e)

def main():
    """
    Main function to handle directory input, load models, and run batch predictions.
    """
    parser = argparse.ArgumentParser(
        description="Run batch predictions on all CSV files in a directory and save results to a CSV."
    )
    parser.add_argument(
        '-d', '--directory',
        required=True,
        help="Path to the directory containing the CSV files to predict."
    )
    parser.add_argument(
        '-o', '--output',
        help="Optional: Path to save the output CSV file. Defaults to 'predictions_[directory_name].csv'."
    )
    args = parser.parse_args()

    # --- Step 1: Validate the input directory ---
    input_dir = args.directory
    if not os.path.isdir(input_dir):
        print(f"\nError: The specified path is not a directory: '{input_dir}'")
        sys.exit(1)

    # --- Step 2: Load the trained model and scaler ---
    try:
        print("\nLoading the trained model and scaler...")
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")
    except Exception as e:
        print(f"\nError: Could not load model or scaler. Make sure '{MODEL_PATH}' and '{SCALER_PATH}' exist.")
        print(f"Details: {e}")
        sys.exit(1)

    # --- Step 3: Find and sort all CSV files ---
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'.")
        sys.exit(0)
        
    csv_files.sort()
    print(f"Found {len(csv_files)} CSV files to process.\n")

    # --- Step 4: Process each file and collect the results ---
    results_list = []
    for filename in tqdm(csv_files, desc="Predicting"):
        full_path = os.path.join(input_dir, filename)
        
        timestamp_str = filename.split('_')[0]
        # Get the actual outcome from the filename for accuracy analysis
        actual_outcome = "WIN" if "win" in filename.lower() else "LOSS"
        
        prediction_score, error = process_single_file(full_path, model, scaler)
        
        if error:
            continue

        threshold = 0.5
        if prediction_score > threshold:
            prediction = "WIN"
            confidence = prediction_score
        else:
            prediction = "LOSS"
            confidence = 1 - prediction_score
            
        results_list.append({
            "timestamp": timestamp_str,
            "actual_outcome": actual_outcome,
            "predicted_outcome": prediction,
            "confidence": f"{confidence:.2%}",
            "raw_score": prediction_score
        })

    # --- Step 5: Save results to a CSV file ---
    if not results_list:
        print("No files were successfully processed. No output file created.")
        return

    results_df = pd.DataFrame(results_list)

    # Determine the output filename
    if args.output:
        output_filename = args.output
    else:
        # Create a default filename based on the input directory
        dir_name = os.path.basename(os.path.normpath(input_dir))
        output_filename = f"predictions_{dir_name}.csv"

    # Save the DataFrame
    results_df.to_csv(output_filename, index=False)
    print("\n" + "="*50)
    print("Batch prediction complete.")
    print(f"Results for {len(results_df)} files have been saved to:")
    print(f"  --> {output_filename}")
    print("="*50)

if __name__ == '__main__':
    main()

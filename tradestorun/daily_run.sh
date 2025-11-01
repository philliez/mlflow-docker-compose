
#!/bin/bash

# --- Configuration ---
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
RCLONE_REMOTE="mega"
REMOTE_DIR="daily_predictions"

# --- Step 1: Fetch the latest data from Phemex ---
echo "Fetching latest data from Phemex..."
# The python script will print the directory it saved the data to
DATA_DIR=$(python3 "$SCRIPT_DIR/fetch_phemex_data.py" | tail -n 1)

if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
    echo "Failed to fetch or find data directory. Exiting."
    exit 1
fi

echo "Data saved to directory: $DATA_DIR"

# --- Step 2: Run batch predictions on the new data ---
echo "Running batch predictions..."
OUTPUT_CSV="predictions_$(basename "$DATA_DIR").csv"
python3 "$SCRIPT_DIR/4_batch_predict.py" -d "$DATA_DIR" -o "$OUTPUT_CSV"

if [ ! -f "$OUTPUT_CSV" ]; then
    echo "Prediction output file not found. Exiting."
    exit 1
fi

echo "Predictions saved to $OUTPUT_CSV"

# --- Step 3: Upload the prediction results to Mega ---
echo "Uploading results to Mega cloud storage..."
rclone copy "$OUTPUT_CSV" "${RCLONE_REMOTE}:${REMOTE_DIR}"

if [ $? -eq 0 ]; then
    echo "Successfully uploaded to Mega."
else
    echo "Failed to upload to Mega."
    exit 1
fi

# --- Step 4: Clean up local files ---
echo "Cleaning up local files..."
rm "$OUTPUT_CSV"
rm -r "$DATA_DIR"

echo "Daily run completed successfully."

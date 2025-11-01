import os
import subprocess
import sys
import pandas as pd
import numpy as np

# Add the current directory to the Python path to import local modules
sys.path.append(os.path.dirname(__file__))

import consolidate_and_process as data_processor
import feature_engineer as fe

import os

import subprocess

import sys

import pandas as pd

import numpy as np

import mlflow



# Add the current directory to the Python path to import local modules

sys.path.append(os.path.dirname(__file__))



import consolidate_and_process as data_processor

import feature_engineer as fe



# --- Configuration ---

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), '2025-06-01') # Example dated directory

EXPECTED_ROWS = 480

EXPECTED_COLS = ['open', 'high', 'low', 'close', 'volume', 'histogram', 'signal', 'macd']

MLFLOW_TRACKING_URI = "http://mlflow:5000"



# --- Set MLflow Tracking URI ---

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)



# --- Step 1: Consolidating and Processing Raw Data ---

print("\n--- Step 1: Consolidating and Processing Raw Data ---")

processed_dfs, labels = data_processor.process_data(BASE_DATA_DIR, EXPECTED_ROWS, EXPECTED_COLS)



if not processed_dfs:

    print("No data to process. Exiting pipeline.")

    sys.exit(1)



# --- Step 2: Apply Feature Engineering ---

print("\n--- Step 2: Applying Feature Engineering ---")

X_data_np, y_data_np, final_feature_cols = fe.apply_feature_engineering(processed_dfs, labels)



if X_data_np is None:

    print("Feature engineering failed or resulted in no data. Exiting pipeline.")

    sys.exit(1)



# --- Step 3: Train Model ---

print("\n--- Step 3: Training Model ---")

# Run 2_train_model.py as a subprocess

train_process = subprocess.run(["python", os.path.join(os.path.dirname(__file__), "2_train_model.py")], capture_output=True, text=True)

print(train_process.stdout)

if train_process.returncode != 0:

    print("Error during model training:")

    print(train_process.stderr)

    sys.exit(1)



# --- Step 4: Run Batch Predictions ---

print("\n--- Step 4: Running Batch Predictions ---")

# Run 4_batch_predict.py as a subprocess

predict_process = subprocess.run(["python", os.path.join(os.path.dirname(__file__), "4_batch_predict.py"), "-d", BASE_DATA_DIR], capture_output=True, text=True)

print(predict_process.stdout)

if predict_process.returncode != 0:

    print("Error during batch prediction:")

    print(predict_process.stderr)

    sys.exit(1)



# --- Step 5: Analyze Predictions ---

print("\n--- Step 5: Analyzing Predictions ---")

# Run analyze_predictions.py as a subprocess

analyze_process = subprocess.run(["python", "/home/phill/Downloads/xgboost/analyze_predictions.py"], capture_output=True, text=True)

print(analyze_process.stdout)

if analyze_process.returncode != 0:

    print("Error during analysis:")

    print(analyze_process.stderr)

    sys.exit(1)



print("\n--- Full Pipeline Execution Complete ---")

# --- Step 1: Consolidating and Processing Raw Data ---
print("\n--- Step 1: Consolidating and Processing Raw Data ---")
processed_dfs, labels = data_processor.process_data(BASE_DATA_DIR, EXPECTED_ROWS, EXPECTED_COLS)

if not processed_dfs:
    print("No data to process. Exiting pipeline.")
    sys.exit(1)

# --- Step 2: Apply Feature Engineering ---
print("\n--- Step 2: Applying Feature Engineering ---")
X_data_np, y_data_np, final_feature_cols = fe.apply_feature_engineering(processed_dfs, labels)

if X_data_np is None:
    print("Feature engineering failed or resulted in no data. Exiting pipeline.")
    sys.exit(1)

# --- Step 3: Train Model ---
print("\n--- Step 3: Training Model ---")
# Run 2_train_model.py as a subprocess
train_process = subprocess.run(["python", os.path.join(os.path.dirname(__file__), "2_train_model.py")], capture_output=True, text=True)
print(train_process.stdout)
if train_process.returncode != 0:
    print("Error during model training:")
    print(train_process.stderr)
    sys.exit(1)

# --- Step 4: Run Batch Predictions ---
print("\n--- Step 4: Running Batch Predictions ---")
# Run 4_batch_predict.py as a subprocess
predict_process = subprocess.run(["python", os.path.join(os.path.dirname(__file__), "4_batch_predict.py"), "-d", BASE_DATA_DIR], capture_output=True, text=True)
print(predict_process.stdout)
if predict_process.returncode != 0:
    print("Error during batch prediction:")
    print(predict_process.stderr)
    sys.exit(1)

# --- Step 5: Analyze Predictions ---
print("\n--- Step 5: Analyzing Predictions ---")
# Run analyze_predictions.py as a subprocess
analyze_process = subprocess.run(["python", "/home/phill/Downloads/xgboost/analyze_predictions.py"], capture_output=True, text=True)
print(analyze_process.stdout)
if analyze_process.returncode != 0:
    print("Error during analysis:")
    print(analyze_process.stderr)
    sys.exit(1)

print("\n--- Full Pipeline Execution Complete ---")

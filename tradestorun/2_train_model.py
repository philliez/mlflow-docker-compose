import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import datetime
import joblib
import mlflow
import mlflow.xgboost
import hashlib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta
import argparse

# --- Configuration ---
X_FILE = 'X_data.npz'
Y_FILE = 'y_data.npz'
MODEL_NAME = 'btcusdt_win_loss_model.joblib'
SCALER_NAME = 'scaler.joblib'

# --- Indicator Configuration ---
def get_enabled_indicators():
    """
    Returns a dictionary of indicators to be used in the model.
    The keys are the names of the indicators and the values are boolean.
    """
    return {
        "rsi": True,
        "stoch": True,
        "adx": True,
        "bbands": True,
    }

def add_indicators(df, enabled_indicators):
    """
    Adds technical indicators to the DataFrame.
    MACD, HISTOGRAM, and SIGNAL are always added.
    """
    df.ta.macd(close=df['close'], fast=12, slow=26, signal=9, append=True)

    if enabled_indicators.get("rsi"):
        df.ta.rsi(close=df['close'], length=14, append=True)
    if enabled_indicators.get("stoch"):
        df.ta.stoch(high=df['high'], low=df['low'], close=df['close'], append=True)
    if enabled_indicators.get("adx"):
        df.ta.adx(high=df['high'], low=df['low'], close=df['close'], length=14, append=True)
    if enabled_indicators.get("bbands"):
        df.ta.bbands(close=df['close'], length=20, append=True)
    
    df.fillna(0, inplace=True)
    return df

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-tracking-uri", type=str, help="MLflow tracking server URI.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow.xgboost.autolog()

    with mlflow.start_run() as run:
        # --- Load Data ---
        print("Loading consolidated data...")
        X = np.load(X_FILE)['data']
        y = np.load(Y_FILE)['data']

        # --- Dataset Tracking ---
        mlflow.log_param("X_FILE", X_FILE)
        mlflow.log_param("Y_FILE", Y_FILE)
        data_hash = hashlib.sha256(X.tobytes()).hexdigest()
        mlflow.log_param("data_hash", data_hash)
        
        # --- Assuming filenames contain dates ---
        # This is a placeholder. You'll need to adapt this to how you actually store date information.
        # For example, if you have a separate file with date ranges, you would load that here.
        # For now, we'll just use the current date as a placeholder.
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=60)
        end_date = today
        mlflow.log_param("start_date", str(start_date))
        mlflow.log_param("end_date", str(end_date))


        # --- Reshape for pandas ---
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        # --- Create DataFrame and add indicators ---
        df = pd.DataFrame(X_reshaped, columns=['open', 'high', 'low', 'close', 'volume'])
        enabled_indicators = get_enabled_indicators()
        df = add_indicators(df, enabled_indicators)
        mlflow.log_param("enabled_indicators", enabled_indicators)
        
        # --- Reshape back to 3D for scaling ---
        X_with_indicators = df.values.reshape(n_samples, n_timesteps, -1)
        
        # --- Split Data ---
        print("Splitting data into training and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(X_with_indicators, y, test_size=0.2, shuffle=False)

        # --- Log dataset parameters ---
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("validation_samples", len(X_val))

        # --- Feature Scaling ---
        n_samples_train, n_timesteps_train, n_features_train = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features_train)
        
        n_samples_val, n_timesteps_val, n_features_val = X_val.shape
        X_val_reshaped = X_val.reshape(-1, n_features_val)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_val_scaled = scaler.transform(X_val_reshaped)
        
        joblib.dump(scaler, SCALER_NAME)
        mlflow.log_artifact(SCALER_NAME)
        print(f"Scaler has been fitted and saved to {SCALER_NAME}")

        # --- Train XGBoost Model ---
        print("Training XGBoost model...")
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1
        )
        
        model.fit(X_train_scaled, y_train.ravel(),
                  early_stopping_rounds=50,
                  eval_set=[(X_val_scaled, y_val.ravel())],
                  verbose=False)
        
        joblib.dump(model, MODEL_NAME)
        print(f"Model saved to {MODEL_NAME}")

        # --- Evaluation and Logging ---
        print("\n--- Evaluating Model ---")
        y_pred = model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Accuracy: {accuracy}")

        # --- Classification Report ---
        report = classification_report(y_val, y_pred, output_dict=True)
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact("classification_report.json")
        print("Classification Report:")
        print(classification_report(y_val, y_pred))

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        print("Confusion matrix saved to confusion_matrix.png")
        
        # --- Feature Importance ---
        fig, ax = plt.subplots(figsize=(12, 8))
        xgb.plot_importance(model, ax=ax)
        plt.title("Feature Importance")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        print("Feature importance plot saved to feature_importance.png")
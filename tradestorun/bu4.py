import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import datetime
import joblib
import os

# --- Configuration ---
X_FILE = 'X_data.npz'
Y_FILE = 'y_data.npz'
SAMPLES_PER_MONTH = 14400  # Approximate samples per month (480 files/day * 30 days)

# --- Load Data ---
print("Loading consolidated data...")
X = np.load(X_FILE)['data']
y = np.load(Y_FILE)['data']

best_accuracy = 0
best_model_name = ""
best_report = ""

for i in range(1, 4):
    num_months = i
    data_size = num_months * SAMPLES_PER_MONTH
    
    if data_size > len(X):
        print(f"Not enough data for {num_months} months. Required: {data_size}, Available: {len(X)}. Skipping.")
        continue

    print(f"\n--- Training model on {num_months} month(s) of data ---")
    
    X_subset = X[:data_size]
    y_subset = y[:data_size]

    MODEL_NAME = f'btcusdt_win_loss_{num_months}m_model.keras'
    SCALER_NAME = f'scaler_{num_months}m.joblib'
    LOG_DIR = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{num_months}m"

    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X_subset, y_subset, test_size=0.2, shuffle=False)

    # --- Feature Scaling ---
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)

    scaler = StandardScaler()
    scaler.fit(X_train_reshaped)
    joblib.dump(scaler, SCALER_NAME)
    print(f"Scaler has been fitted and saved to {SCALER_NAME}")

    X_train_scaled = scaler.transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)

    # --- Handle Class Imbalance ---
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    # --- Build the CNN-LSTM Model ---
    model = Sequential([
        Input(shape=(n_timesteps, n_features)),
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # --- Callbacks and Training ---
    callbacks = [
        ModelCheckpoint(MODEL_NAME, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    ]

    print(f"--- Starting Model Training for {num_months} month(s) ---")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_val_scaled, y_val),
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # --- Evaluation and Reporting ---
    val_accuracy = max(history.history['val_accuracy'])
    report = f"Model: {MODEL_NAME}\n"
    report += f"Data: {num_months} month(s)\n"
    report += f"Validation Accuracy: {val_accuracy:.4f}\n"

    report_filename = f"report_{num_months}m.txt"
    with open(report_filename, "w") as f:
        f.write(report)
    print(f"Report saved to {report_filename}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model_name = MODEL_NAME
        best_report = report

# --- Final Report ---
final_report = f"\n--- Best Model ---\n{best_report}"
print(final_report)
with open("best_model_report.txt", "w") as f:
    f.write(final_report)

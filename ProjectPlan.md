## Project Plan: XGBoost-based Trading Models

This plan outlines the steps to develop and track XGBoost-based trading models for creating Freqtrade strategies.

- [ ] **1. Unzip the data:** The first step is to unzip the `2_month_data.zip` file to access the raw data.
- [ ] **2. Consolidate and process data:** Create a script to consolidate the data from June, July, and August, and then process it into a single, usable format.
- [ ] **3. Enhance Feature Engineering:** Improve the `feature_engineer.py` script to create more informative features.
- [ ] **4. Update Prediction Script:** Adjust the `3_predict_and_evaluate.py` script to work with the new XGBoost model and features.
- [ ] **5. Run the pipeline:** Execute the complete pipeline to train the new model and generate predictions.
- [ ] **6. Evaluate and commit:** Analyze the results and, if satisfactory, commit the changes to version control.

## Model Training

- [x] **Switch to XGBoost:** The training script `2_train_model.py` has been updated to use `xgboost.XGBClassifier` instead of TensorFlow/Keras.
- [x] **Indicator Calculation:** The training script now uses `pandas-ta` to calculate technical indicators. `MACD`, `HISTOGRAM`, and `SIGNAL` are always included, and other indicators can be enabled/disabled.

## MLflow Enhancements

- [x] **Enhanced MLflow Tracking:** The training script has been updated to log:
    - Dataset information (file names, hash, start/end dates).
    - Enabled indicators.
    - XGBoost-specific parameters.
    - Feature importance plots.
    - Classification reports and confusion matrices.

## Utility Scripts

- [x] **Create `run_mlflow_server.py`:** A script to start the MLflow UI without Docker Compose.

## Colab Integration

- [x] **Create `COLAB_INSTRUCTIONS.md`:** Instructions on how to connect a Google Colab notebook to the local MLflow server using ngrok.


## Project Plan: XGBoost-based Trading Models

This plan outlines the steps to develop and track XGBoost-based trading models for creating Freqtrade strategies.

- [ ] **1. Enhance Feature Engineering:** Improve the `feature_engineer.py` script to create more informative features.
- [ ] **2. Update Prediction Script:** Adjust the `3_predict_and_evaluate.py` script to work with the new XGBoost model and features.
- [ ] **3. Run the pipeline:** Execute the complete pipeline to train the new model and generate predictions.
- [ ] **4. Evaluate and commit:** Analyze the results and, if satisfactory, commit the changes to version control.

## Data Management

- [x] **Compress OHLC Data:** All OHLC CSV data has been compressed into `ohlc_data.zip` to reduce the repository size.
- [x] **Update Data Processing Script:** The `1_consolidate_and_process.py` script has been updated to read data directly from the `ohlc_data.zip` file.

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
    - [x] **Add `--host` argument:** The script now accepts a `--host` argument to allow connections from ngrok.
- [x] **Create `setup_venv.sh`:** A script to automate the creation of the virtual environment and installation of dependencies.

## Colab Integration

- [x] **Update `COLAB_INSTRUCTIONS.md`:** The instructions have been updated with the user's ngrok URL and authtoken.
- [x] **Create `colab_runner.ipynb`:** A Colab notebook has been created to run the training pipeline, with checks for mounted drive and installed dependencies.

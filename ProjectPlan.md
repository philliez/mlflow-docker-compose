## Project Plan: Improve Feature Engineering and Data Diversity

This plan outlines the steps to enhance the feature engineering process and incorporate more diverse data for training and testing the model. The goal is to improve model performance and robustness.

- [ ] **1. Unzip the data:** The first step is to unzip the `2_month_data.zip` file to access the raw data.
- [ ] **2. Consolidate and process data:** Create a script to consolidate the data from June, July, and August, and then process it into a single, usable format.
- [ ] **3. Enhance Feature Engineering:** Improve the `feature_engineer.py` script to create more informative features.
- [ ] **4. Update Training Script:** Modify the `2_train_model.py` script to use the new engineered features and save the updated model.
- [ ] **5. Update Prediction Script:** Adjust the `3_predict_and_evaluate.py` script to work with the new model and features.
- [ ] **6. Run the pipeline:** Execute the complete pipeline to train the new model and generate predictions.
- [ ] **7. Evaluate and commit:** Analyze the results and, if satisfactory, commit the changes to version control.

## Utility Scripts

- [x] **Create `run_mlflow_server.py`:** A script to start the MLflow UI without Docker Compose.

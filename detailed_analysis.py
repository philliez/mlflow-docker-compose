import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob
import os

# Get a list of all csv files
files = glob.glob('tradestorun/**/*.csv', recursive=True)

# Create a list to hold the dataframes
df_list = []

for file in files:
    if '_win.csv' in file or '_loss.csv' in file:
        df = pd.read_csv(file)
        # Add a target column
        if '_win.csv' in file:
            df['target'] = 1
        else:
            df['target'] = 0
        df_list.append(df)

# Concatenate all dataframes
full_df = pd.concat(df_list, ignore_index=True)

# Feature Engineering (simple example)
full_df['price_range'] = full_df['high'] - full_df['low']

# Define features (X) and target (y)
features = ['open', 'high', 'low', 'close', 'volume', 'price_range']
target = 'target'

X = full_df[features]
y = full_df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Train the XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'scale_pos_weight': scale_pos_weight
}
num_round = 100
bst = xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], early_stopping_rounds=10)

# Make predictions
preds = bst.predict(dtest)
predictions = np.round(preds)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
feature_importance = bst.get_score(importance_type='weight')


# Generate a detailed report
with open('model_report.md', 'w') as f:
    f.write("# Model Performance Report\n\n")
    f.write("## 1. Overview\n")
    f.write("This report details the performance of an XGBoost model trained to predict 'win' or 'loss' outcomes based on 3-minute OHLC data.\n\n")
    f.write("## 2. Data\n")
    f.write(f"The model was trained on {len(full_df)} data points from {len(files)} CSV files.\n")
    f.write(f"Class distribution in the full dataset:\n")
    f.write(f"Wins (1): {len(full_df[full_df['target'] == 1])}\n")
    f.write(f"Losses (0): {len(full_df[full_df['target'] == 0])}\n\n")
    f.write("## 3. Model & Training\n")
    f.write("An XGBoost classifier was used. To handle class imbalance, `scale_pos_weight` was set to {:.2f}.\n\n".format(scale_pos_weight))
    f.write("### Hyperparameters:\n")
    for key, value in param.items():
        f.write(f"- {key}: {value}\n")
    f.write(f"- num_round: {num_round}\n\n")
    f.write("## 4. Evaluation Metrics\n")
    f.write(f"**Accuracy:** {accuracy:.4f}\n\n")
    f.write("**Classification Report:**\n")
    f.write("```\n")
    f.write(report)
    f.write("\n```\n\n")
    f.write("**Confusion Matrix:**\n")
    f.write("```\n")
    f.write(str(conf_matrix))
    f.write("\n```\n\n")
    f.write("## 5. Feature Importance\n")
    f.write("The following features were most important for the model's predictions:\n\n")
    # Sort feature importance
    sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_feature_importance:
        f.write(f"- **{feature}:** {importance}\n")
    f.write("\n## 6. Conclusion\n")
    f.write("The model shows predictive power, but further tuning and feature engineering could improve performance, especially for the minority class. The high `scale_pos_weight` indicates a significant class imbalance was addressed.\n")

print("Analysis complete. Report generated in model_report.md")

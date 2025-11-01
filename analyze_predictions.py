import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_predictions(file_path):
    df = pd.read_csv(file_path)

    # Convert 'confidence' to numeric, handling the '%' sign
    df['confidence_numeric'] = df['confidence'].str.replace('%', '').astype(float) / 100

    # Convert 'actual_outcome' and 'predicted_outcome' to binary (1 for WIN, 0 for LOSS)
    # Assuming 'WIN' is the positive class
    df['actual_binary'] = df['actual_outcome'].apply(lambda x: 1 if x == 'WIN' else 0)
    df['predicted_binary'] = df['predicted_outcome'].apply(lambda x: 1 if x == 'WIN' else 0)

    # Calculate metrics
    accuracy = accuracy_score(df['actual_binary'], df['predicted_binary'])
    precision_win = precision_score(df['actual_binary'], df['predicted_binary'], pos_label=1, zero_division=0)
    recall_win = recall_score(df['actual_binary'], df['predicted_binary'], pos_label=1, zero_division=0)
    f1_win = f1_score(df['actual_binary'], df['predicted_binary'], pos_label=1, zero_division=0)

    precision_loss = precision_score(df['actual_binary'], df['predicted_binary'], pos_label=0, zero_division=0)
    recall_loss = recall_score(df['actual_binary'], df['predicted_binary'], pos_label=0, zero_division=0)
    f1_loss = f1_score(df['actual_binary'], df['predicted_binary'], pos_label=0, zero_division=0)

    report_output = []
    report_output.append("\n--- Prediction Analysis Report ---")
    report_output.append(f"Total Predictions: {len(df)}")
    report_output.append(f"Overall Accuracy: {accuracy:.2%}")
    report_output.append("\nMetrics for 'WIN' (Positive Class):")
    report_output.append(f"  Precision: {precision_win:.2%}")
    report_output.append(f"  Recall: {recall_win:.2%}")
    report_output.append(f"  F1-Score: {f1_win:.2%}")
    report_output.append("\nMetrics for 'LOSS' (Negative Class):")
    report_output.append(f"  Precision: {precision_loss:.2%}")
    report_output.append(f"  Recall: {recall_loss:.2%}")
    report_output.append(f"  F1-Score: {f1_loss:.2%}")
    report_output.append("\n----------------------------------")

    # Further investigation: Check if the thresholding is the issue
    misclassified_wins = df[(df['actual_outcome'] == 'WIN') & (df['predicted_outcome'] == 'LOSS')]
    if not misclassified_wins.empty:
        report_output.append("\n--- Misclassified 'WIN' trades (predicted as 'LOSS') ---")
        report_output.append(misclassified_wins[['timestamp', 'raw_score', 'confidence']].to_string())
        report_output.append("--------------------------------------------------------")

    # --- Generate and Save Charts ---
    output_dir = os.path.dirname(file_path)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(df['actual_binary'], df['predicted_binary'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['LOSS', 'WIN'], yticklabels=['LOSS', 'WIN'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    report_output.append(f"\nConfusion Matrix saved to: {cm_path}")

    # 2. Distribution of Raw Scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['actual_outcome'] == 'WIN']['raw_score'], color='green', kde=True, label='Actual WIN', stat='density')
    sns.histplot(df[df['actual_outcome'] == 'LOSS']['raw_score'], color='red', kde=True, label='Actual LOSS', stat='density')
    plt.axvline(0.5, color='blue', linestyle='--', label='Prediction Threshold (0.5)')
    plt.title('Distribution of Raw Prediction Scores')
    plt.xlabel('Raw Score (Probability of WIN)')
    plt.ylabel('Density')
    plt.legend()
    score_dist_path = os.path.join(output_dir, 'raw_score_distribution.png')
    plt.savefig(score_dist_path)
    plt.close()
    report_output.append(f"Raw Score Distribution saved to: {score_dist_path}")

    # --- Save Text Report ---
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        for line in report_output:
            f.write(line + '\n')
    print(f"\nFull analysis report (text) saved to: {report_path}")

    # Print the report to console as well
    for line in report_output:
        print(line)

if __name__ == '__main__':
    predictions_file = "/home/phill/Downloads/xgboost/predictions_2025-06-01.csv"
    analyze_predictions(predictions_file)

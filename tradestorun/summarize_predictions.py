import pandas as pd
import numpy as np

file_path = 'predictions_daily_backtest_output.csv'
df = pd.read_csv(file_path)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

df['correct_prediction'] = df['actual_outcome'] == df['predicted_outcome']

daily_accuracy = df.groupby('date')['correct_prediction'].mean().reset_index()
daily_accuracy.rename(columns={'correct_prediction': 'accuracy'}, inplace=True)

overall_accuracy = df['correct_prediction'].mean()

avg_confidence_correct = df[df['correct_prediction']]['confidence'].str.rstrip('%').astype('float').mean()
avg_confidence_incorrect = df[~df['correct_prediction']]['confidence'].str.rstrip('%').astype('float').mean()

print("Daily Prediction Accuracy:")
for index, row in daily_accuracy.iterrows():
    print(f"- {row['date']}: {row['accuracy']:.2%}")

print("\nOverall Prediction Accuracy:")
print(f"- {overall_accuracy:.2%}")

print("\nAverage Confidence:")
print(f"- Correct Predictions: {avg_confidence_correct:.2f}%")
print(f"- Incorrect Predictions: {avg_confidence_incorrect:.2f}%")

# Logarithmic improvement calculation (using confidence)
# We can think of "log improve" as the log-likelihood ratio.
# For simplicity, we'll calculate the log of the ratio of average confidences.
log_improve = np.log(avg_confidence_correct / avg_confidence_incorrect)

print(f"\nLogarithmic Improvement (based on confidence): {log_improve:.4f}")

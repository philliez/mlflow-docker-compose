
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import os

def summarize_predictions(file_path):
    """
    Summarizes prediction data from a CSV file, calculating daily accuracy and confidence metrics.
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['correct_prediction'] = df['actual_outcome'] == df['predicted_outcome']

    # Calculate daily summaries
    daily_summary = df.groupby('date').agg(
        total_predictions=('timestamp', 'count'),
        correct_predictions=('correct_prediction', lambda x: x.sum()),
        accuracy=('correct_prediction', 'mean')
    ).reset_index()

    # Format accuracy as percentage
    daily_summary['accuracy'] = daily_summary['accuracy'].apply(lambda x: f"{x:.2%}")

    return daily_summary

def upload_to_google_sheet(summary_df, sheet_name, credentials_path, share_with):
    """
    Uploads a DataFrame to a specified Google Sheet, sharing it with the given email addresses.
    """
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
        client = gspread.authorize(creds)

        # Create or open the spreadsheet
        try:
            spreadsheet = client.open(sheet_name)
        except gspread.exceptions.SpreadsheetNotFound:
            spreadsheet = client.create(sheet_name)

        # Share the spreadsheet with specified users as editors
        for email in share_with:
            spreadsheet.share(email, perm_type='user', role='writer')

        # Select the first worksheet and clear it
        worksheet = spreadsheet.sheet1
        worksheet.clear()

        # Write the DataFrame to the worksheet
        worksheet.update([summary_df.columns.values.tolist()] + summary_df.values.tolist())
        print(f"Successfully uploaded data to '{sheet_name}'.")
        print(f"Shared with: {', '.join(share_with)}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your credentials file is correct and the Google Sheets API is enabled.")

def main():
    """
    Main function to orchestrate the summarization and upload process.
    """
    # --- Configuration ---
    PREDICTIONS_FILE = 'predictions_daily_backtest_output.csv'
    GOOGLE_SHEET_NAME = 'Daily Prediction Summary'
    CREDENTIALS_FILE = 'google_credentials.json'  # <-- IMPORTANT: Path to your credentials file
    SHARE_WITH_EMAILS = ['svenceelen@gmail.com', 'phillh@gmail.com']

    # Check if the predictions file exists
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"Error: The file '{PREDICTIONS_FILE}' was not found.")
        return

    # Check if the credentials file exists
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"Error: The Google credentials file '{CREDENTIALS_FILE}' was not found.")
        print("Please follow the instructions to create it and place it in the same directory.")
        return

    # --- Run the process ---
    summary_df = summarize_predictions(PREDICTIONS_FILE)
    upload_to_google_sheet(summary_df, GOOGLE_SHEET_NAME, CREDENTIALS_FILE, SHARE_WITH_EMAILS)

if __name__ == '__main__':
    main()


import ccxt
import pandas as pd
from datetime import datetime
import os

def fetch_phemex_data():
    """
    Fetches the latest 480 3-minute candles for BTC/USDT from Phemex
    and saves them to a CSV file in a date-stamped directory.
    """
    try:
        phemex = ccxt.phemex()
        
        # Fetch OHLCV data
        ohlcv = phemex.fetch_ohlcv('BTC/USDT', '3m', limit=480)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Create directory for the current date
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        if not os.path.exists(today_str):
            os.makedirs(today_str)
            
        # Save to CSV
        # The filename will be the timestamp of the latest candle
        latest_timestamp_str = df['timestamp'].iloc[-1].strftime('%Y-%m-%dT%H-%M-%S')
        output_filename = os.path.join(today_str, f"{latest_timestamp_str}_phemex_data.csv")
        
        df.to_csv(output_filename, index=False)
        
        print(f"Successfully fetched and saved 480 candles to '{output_filename}'")
        print(today_str) # Print the directory name for the next script

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        exit(1)

if __name__ == '__main__':
    fetch_phemex_data()

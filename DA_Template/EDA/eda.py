import pandas as pd
import os

def load_data(path, files_dict):
    df = pd.DataFrame()
    for platform, file_list in files_dict.items():
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            if not os.path.exists(file_path):
                print(f"Warning: File not found - {file_path}")
                continue
            temp_df = pd.read_csv(file_path)
            temp_df['platform'] = platform
            df = pd.concat([df, temp_df], ignore_index=True)
    return df

def filter_data_by_date(df, start_date, end_date):
    # Convert Streamlit date inputs to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Ensure the 'datetime' column is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # Convert 'datetime' to UTC-6
    df['datetime'] = pd.DatetimeIndex(df['datetime']).tz_localize('UTC').tz_convert('America/Mexico_City').tz_localize(None)
    
    # Filter data within the date range
    return df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

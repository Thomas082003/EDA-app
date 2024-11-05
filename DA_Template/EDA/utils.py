import pandas as pd
from PIL import Image
from io import BytesIO
import requests

def load_data(path, filesDict):
    df = pd.DataFrame()
    for platform, file_list in filesDict.items():
        for file_name in file_list:
            temp_df = pd.read_csv(f"{path}/{file_name.lstrip('data/')}")
            temp_df['platform'] = platform
            df = pd.concat([df, temp_df], ignore_index=True)
    return df

def selectDataByDate(df, start_date, end_date):
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    filtered_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    return filtered_df


import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['weekday'] = df['time'].dt.weekday < 5
    df['day_of_week'] = df['time'].dt.dayofweek
    return df

def save_data(df, output_file_path):
    df.to_csv(output_file_path, index=False)

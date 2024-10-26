import pandas as pd

def prepare_data(data_path):
    """Prepare and enhance the dataset with time-based features."""
    data = pd.read_csv(data_path)
    data['time'] = pd.to_datetime(data['time'])
    data['day_of_week'] = data['time'].dt.day_name()
    data['month'] = data['time'].dt.month
    data['day'] = data['time'].dt.day
    data['weekday'] = data['time'].dt.weekday < 5
    return data

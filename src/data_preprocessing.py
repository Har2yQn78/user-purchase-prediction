import pandas as pd

def preprocess_train_data(train_file):
    train_data = pd.read_csv(train_file)
    train_data['time'] = pd.to_datetime(train_data['time'])
    train_data['day_of_week'] = train_data['time'].dt.day_name()
    train_data['month'] = train_data['time'].dt.month
    train_data['day'] = train_data['time'].dt.day
    train_data['weekday'] = train_data['time'].dt.weekday < 5
    output_file_path = 'filtered_enhanced_train.csv'
    train_data.to_csv(output_file_path, index=False)
    return output_file_path

def preprocess_test_data(test_file):
    test_data = pd.read_csv(test_file)
    test_data['time'] = pd.to_datetime(test_data['time'])
    test_data['day_of_week'] = test_data['time'].dt.day_name()
    test_data['month'] = test_data['time'].dt.month
    test_data['day'] = test_data['time'].dt.day
    test_data['weekday'] = test_data['time'].dt.weekday < 5
    return test_data

if __name__ == "__main__":
    preprocess_train_data('train.csv')

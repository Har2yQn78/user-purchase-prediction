import pandas as pd

def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(['user', 'bought', 'time'], axis=1)
    y = data['bought']
    X = pd.get_dummies(X, columns=['day_of_week'], drop_first=True)
    X = X.astype(float)
    return X, y

if __name__ == "__main__":
    X, y = load_and_prepare_data('filtered_enhanced_train.csv')

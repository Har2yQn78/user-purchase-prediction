import pandas as pd
import joblib

def add_features(df):
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['weekday'] = df['time'].dt.weekday < 5
    df['day_of_week'] = df['time'].dt.dayofweek
    return df

def load_models(model_file):
    return joblib.load(model_file)

def predict(user_models, test_data):
    predictions = []

    for user in test_data['user'].unique():
        user_test_data = test_data[test_data['user'] == user]
        X_test = user_test_data.drop(['user', 'time'], axis=1)
        X_test = pd.get_dummies(X_test, columns=['day_of_week'], drop_first=True)

        if user in user_models:
            user_model = user_models[user]
            model_feature_names = user_model.get_booster().feature_names
            X_test = X_test.reindex(columns=model_feature_names, fill_value=0)

            user_predictions = user_model.predict(X_test)
            predictions.extend(user_predictions)
        else:
            predictions.extend([0] * len(user_test_data))

    test_data['bought'] = predictions
    return test_data

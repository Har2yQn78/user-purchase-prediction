# src/predict.py
import pandas as pd
import numpy as np
from joblib import load
from src.utils.data_preprocessing import prepare_data

def make_predictions(test_data_path, model_path='user_ensemble_models.pkl'):
    user_ensemble_models = load(model_path)
    test_data = prepare_data(test_data_path)

    user_time_data = test_data[['user', 'time']].copy()
    test_data = pd.get_dummies(test_data.drop(['user', 'time'], axis=1), columns=['day_of_week'], drop_first=True)
    test_data = test_data.astype(float)

    predictions = []
    for user in test_data['user'].unique():
        user_test_data = test_data[test_data['user'] == user].copy().drop('user', axis=1, errors='ignore')

        if user in user_ensemble_models:
            ensemble = user_ensemble_models[user]
            predictions.extend(ensemble.predict(user_test_data))
        else:
            predictions.extend([0] * len(user_test_data))

    user_time_data['bought'] = predictions
    user_time_data.to_csv('ensemble_predictions.csv', index=False)

import pandas as pd
import numpy as np
import joblib
from utils.data_preprocessing import prepare_data

user_ensemble_models = joblib.load('models/user_ensemble_models.pkl')
test_data = prepare_data('data/test.csv')
user_time_data = test_data[['user', 'time']].copy()
test_data = pd.get_dummies(test_data.drop(['time'], axis=1), columns=['day_of_week'], drop_first=True).astype(float)

predictions = []
for user in test_data['user'].unique():
    user_test_data = test_data[test_data['user'] == user].copy().drop('user', axis=1, errors='ignore')
    if user in user_ensemble_models:
        ensemble = user_ensemble_models[user]
        user_predictions = ensemble.predict(user_test_data)
        predictions.extend(user_predictions)
    else:
        predictions.extend([0] * len(user_test_data))

user_time_data['bought'] = predictions
user_time_data.to_csv('ensemble_predictions.csv', index=False)

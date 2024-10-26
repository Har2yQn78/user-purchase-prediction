import joblib
import pandas as pd
from utils.data_preprocessing import prepare_data
from ensemble_model import EnsembleModel

train_data = prepare_data('data/train.csv')
user_ensemble_models = {}

for user in train_data['user'].unique():
    user_data = train_data[train_data['user'] == user]
    if len(user_data) < 5:
        continue

    X_user = user_data.drop(['user', 'bought', 'time'], axis=1)
    y_user = user_data['bought']
    X_user = pd.get_dummies(X_user, columns=['day_of_week'], drop_first=True).astype(float)

    ensemble = EnsembleModel(user)
    ensemble.train(X_user, y_user)
    user_ensemble_models[user] = ensemble

joblib.dump(user_ensemble_models, 'models/user_ensemble_models.pkl')

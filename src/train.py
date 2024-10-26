# src/train.py
import pandas as pd
from src.utils.data_preprocessing import prepare_data
from src.ensemble_model import EnsembleModel
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import dump

def train_models(data_path):
    train_data = prepare_data(data_path)
    user_ensemble_models = {}

    for user in train_data['user'].unique():
        user_data = train_data[train_data['user'] == user]

        if len(user_data) < 5:
            continue

        X_user = pd.get_dummies(user_data.drop(['user', 'bought', 'time'], axis=1), columns=['day_of_week'], drop_first=True)
        y_user = user_data['bought']

        models = {
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'random_forest': RandomForestClassifier(class_weight='balanced'),
            'catboost': CatBoostClassifier(verbose=False),
            'lightgbm': LGBMClassifier()
        }

        ensemble = EnsembleModel(user, models)
        ensemble.train(X_user, y_user)
        user_ensemble_models[user] = ensemble

    dump(user_ensemble_models, 'user_ensemble_models.pkl')

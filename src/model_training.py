import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib


def train_user_models(data, output_file):
    user_models = {}

    neg_count = len(data[data['bought'] == 0])
    pos_count = len(data[data['bought'] == 1])
    scale_pos_weight = neg_count / pos_count

    for user in data['user'].unique():
        user_data = data[data['user'] == user]

        X = user_data.drop(['user', 'bought'], axis=1)
        y = user_data['bought']

        if len(y) < 5:
            continue

        X = pd.get_dummies(X, columns=['time'], drop_first=True)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        print(f'User: {user}, F1 Score: {f1}')

        user_models[user] = xgb_model

    joblib.dump(user_models, output_file)


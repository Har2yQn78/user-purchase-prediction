import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import joblib
import pandas as pd
from feature_engineering import load_and_prepare_data


def train_model(X, y):
    param_dist = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 12),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'scale_pos_weight': uniform(1, 5),
        'max_delta_step': randint(0, 10),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=100,
        scoring='f1',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    user_models = {}
    data = pd.read_csv('filtered_enhanced_train.csv')
    for user in data['user'].unique():
        user_data = data[data['user'] == user]

        if len(user_data) < 5:
            continue

        X_user = user_data.drop(['user', 'bought', 'time'], axis=1)
        y_user = user_data['bought']
        X_user = pd.get_dummies(X_user, columns=['day_of_week'], drop_first=True)
        X_user = X_user.astype(float)
        random_search.fit(X_user, y_user)
        user_models[user] = random_search.best_estimator_
        print(
            f'User: {user}, Best parameters: {random_search.best_params_}, Best F1-score: {random_search.best_score_}')

    joblib.dump(user_models, 'user_models.pkl')


if __name__ == "__main__":
    X, y = load_and_prepare_data('filtered_enhanced_train.csv')
    train_model(X, y)

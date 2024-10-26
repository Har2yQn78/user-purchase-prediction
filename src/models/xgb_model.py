import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

xgb_param_dist = {
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

def train_xgb(X_train, y_train, cv=3, n_iter=100):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    random_search = RandomizedSearchCV(
        model,
        param_distributions=xgb_param_dist,
        n_iter=n_iter,
        scoring='f1',
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

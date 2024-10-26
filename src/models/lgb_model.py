from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

lgb_param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 12),
    'learning_rate': uniform(0.01, 0.3),
    'num_leaves': randint(20, 100),
    'subsample': uniform(0.6, 1.0),
    'colsample_bytree': uniform(0.6, 1.0),
    'min_child_samples': randint(1, 50),
    'min_split_gain': uniform(0, 1),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

def train_lgb(X_train, y_train, cv=3, n_iter=100):
    model = LGBMClassifier()
    random_search = RandomizedSearchCV(
        model,
        param_distributions=lgb_param_dist,
        n_iter=n_iter,
        scoring='f1',
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

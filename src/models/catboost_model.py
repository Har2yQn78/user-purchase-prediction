from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

catboost_param_dist = {
    'iterations': randint(100, 1000),
    'depth': randint(3, 12),
    'learning_rate': uniform(0.01, 0.3),
    'l2_leaf_reg': uniform(1, 10),
    'rsm': uniform(0.6, 0.4),
    'subsample': uniform(0.6, 0.4),
    'random_strength': uniform(0, 1)
}

def train_catboost(X_train, y_train, cv=3, n_iter=100):
    model = CatBoostClassifier(verbose=False)
    random_search = RandomizedSearchCV(
        model,
        param_distributions=catboost_param_dist,
        n_iter=n_iter,
        scoring='f1',
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

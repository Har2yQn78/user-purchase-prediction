from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

rf_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 12),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False],
    'min_impurity_decrease': uniform(0, 0.1)
}

def train_rf(X_train, y_train, cv=3, n_iter=100):
    model = RandomForestClassifier(class_weight='balanced')
    random_search = RandomizedSearchCV(
        model,
        param_distributions=rf_param_dist,
        n_iter=n_iter,
        scoring='f1',
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

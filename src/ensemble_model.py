import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from utils.model_utils import xgb_param_dist, rf_param_dist, catboost_param_dist, lgb_param_dist
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

class EnsembleModel:
    def __init__(self, user_id):
        self.user_id = user_id
        self.models = {
            'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'random_forest': RandomForestClassifier(class_weight='balanced'),
            'catboost': CatBoostClassifier(verbose=False),
            'lightgbm': LGBMClassifier()
        }
        self.param_distributions = {
            'xgboost': xgb_param_dist,
            'random_forest': rf_param_dist,
            'catboost': catboost_param_dist,
            'lightgbm': lgb_param_dist
        }
        self.best_models = {}
        self.model_weights = {}  # Store F1 scores as weights for this specific user

    def train(self, X, y, cv=3, n_iter=100):
        """Train all models using RandomizedSearchCV and store their F1 scores for this user."""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        for name, model in self.models.items():
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.param_distributions[name],
                n_iter=n_iter,
                scoring='f1',
                cv=cv,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )
            random_search.fit(X_train, y_train)
            self.best_models[name] = random_search.best_estimator_

            val_pred = random_search.best_estimator_.predict(X_val)
            f1 = f1_score(y_val, val_pred)
            self.model_weights[name] = f1

        weight_sum = sum(self.model_weights.values())
        self.model_weights = {k: v / weight_sum for k, v in self.model_weights.items()}

    def predict(self, X):
        """Make predictions using weighted voting based on user-specific F1 scores."""
        probabilities = np.zeros((len(X), 2))
        for name, model in self.best_models.items():
            if hasattr(model, 'predict_proba'):
                model_prob = model.predict_proba(X)
                probabilities += model_prob * self.model_weights[name]
            else:
                pred = model.predict(X)
                probabilities[:, 1] += (pred * self.model_weights[name])
                probabilities[:, 0] += ((1 - pred) * self.model_weights[name])

        return (probabilities[:, 1] > 0.5).astype(int)

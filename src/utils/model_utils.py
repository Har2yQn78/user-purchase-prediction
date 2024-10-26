from scipy.stats import uniform, randint

# Parameter distributions for model tuning
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

rf_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 12),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False],
    'min_impurity_decrease': uniform(0, 0.1)
}

catboost_param_dist = {
    'iterations': randint(100, 1000),
    'depth': randint(3, 12),
    'learning_rate': uniform(0.01, 0.3),
    'l2_leaf_reg': uniform(1, 10),
    'rsm': uniform(0.6, 0.4),
    'subsample': uniform(0.6, 0.4),
    'random_strength': uniform(0, 1)
}

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
    'reg_lambda': uniform(0, 1),
}

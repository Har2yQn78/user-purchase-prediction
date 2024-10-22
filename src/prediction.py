import pandas as pd
import joblib
from data_preprocessing import preprocess_test_data

def make_predictions(test_file, model_file):
    test_data = preprocess_test_data(test_file)
    user_time_data = test_data[['user', 'time']].copy()
    test_data = test_data.drop(['time'], axis=1)
    test_data = pd.get_dummies(test_data, columns=['day_of_week'], drop_first=True)
    test_data = test_data.astype(float)

    user_models = joblib.load(model_file)
    predictions = []

    for user in test_data['user'].unique():
        user_test_data = test_data[test_data['user'] == user].copy()

        if user in user_models:
            user_model = user_models[user]
            model_feature_names = user_model.get_booster().feature_names
            user_test_data = user_test_data.reindex(columns=model_feature_names, fill_value=0)
            if 'user' in user_test_data.columns:
                user_test_data = user_test_data.drop(columns=['user'], axis=1)
            user_predictions = user_model.predict(user_test_data)
            predictions.extend(user_predictions)
        else:
            predictions.extend([0] * len(user_test_data))

    user_time_data['bought'] = predictions
    output_file_path = 'predictions_output.csv'
    user_time_data.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    make_predictions('test.csv', 'user_models.pkl')

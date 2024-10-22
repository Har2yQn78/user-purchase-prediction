from data_preprocessing import preprocess_train_data
from feature_engineering import load_and_prepare_data
from model_training import train_model
from prediction import make_predictions

if __name__ == "__main__":
    train_file = preprocess_train_data('train.csv')
    X, y = load_and_prepare_data(train_file)
    train_model(X, y)
    make_predictions('test.csv', 'user_models.pkl')

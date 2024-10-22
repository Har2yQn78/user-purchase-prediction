import pandas as pd
from data_preprocessing import load_data, preprocess_data
from model_training import train_user_models
from prediction import predict

def main():
    # Load and preprocess training data
    train_data = load_data('data/train.csv')
    train_data = preprocess_data(train_data)

    # Train the model
    train_user_models(train_data)

    # Load and preprocess test data
    test_data = load_data('data/test.csv')
    test_data = preprocess_data(test_data)

    # Make predictions
    predictions = predict(test_data)
    output_file_path = 'data/predictions_output.csv'
    predictions.to_csv(output_file_path, index=False)
    print(f"Predictions saved to {output_file_path}")

if __name__ == "__main__":
    main()

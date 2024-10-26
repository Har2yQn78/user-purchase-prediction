# main.py
import argparse
from src.train import train_models
from src.predict import make_predictions

def main():
    parser = argparse.ArgumentParser(description="Ensemble Model Trainer and Predictor")
    parser.add_argument('--train', action='store_true', help="Train the models on train data")
    parser.add_argument('--predict', action='store_true', help="Predict using trained models on test data")
    parser.add_argument('--train_data', type=str, default="data/train.csv", help="Path to training data")
    parser.add_argument('--test_data', type=str, default="data/test.csv", help="Path to test data")
    args = parser.parse_args()

    if args.train:
        train_models(args.train_data)
        print("Training complete.")

    if args.predict:
        make_predictions(args.test_data)
        print("Predictions saved to ensemble_predictions.csv")

if __name__ == "__main__":
    main()

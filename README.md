# Ensemble User Prediction Project

This project is designed to predict user behavior using an ensemble of models. It integrates four machine learning classifiers: **XGBoost**, **Random Forest**, **CatBoost**, and **LightGBM**, each optimized and weighted individually per user based on F1 scores.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset Requirements](#dataset-requirements)
5. [Project Details](#project-details)

---

## Project Structure

The project follows a modular structure for organization and maintainability:

```plaintext
project/
│
├── data/
│   ├── train.csv               # Training data file (not included in repo)
│   └── test.csv                # Test data file (not included in repo)
│
├── models/                     # Model definitions and configurations
│   ├── xgb_model.py            # XGBoost model setup and tuning
│   ├── rf_model.py             # Random Forest model setup and tuning
│   ├── catboost_model.py       # CatBoost model setup and tuning
│   └── lgb_model.py            # LightGBM model setup and tuning
│
├── src/
│   ├── model_utils.py          # Ensemble model orchestration, training, and weighting
│   ├── data_utils.py           # Data preparation and feature engineering
│   └── predict.py              # Generates predictions using the trained models
│
├── ensemble_predictions.csv    # Prediction output (generated after running predictions)
├── main.py                     # Main script to initiate training or prediction
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

# Installation
1. Clone this repository:
```bash
git clone https://github.com/your_username/ensemble-user-prediction.git
cd ensemble-user-prediction
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

# Usage
The project uses main.py as the entry point. The script provides options to either train models or generate predictions based on trained models.

# Training the Models
To train models for each user in the dataset, use the --train flag with the path to your training data file.
```bash
python main.py --train --train_data=data/train.csv
```
This command:

1. Loads the training data from data/train.csv.
2. Prepares the data with time-based features.
3. Trains and tunes each model for every unique user in the data.
4. Saves the trained models to user_ensemble_models.pkl.

# Making Predictions
To generate predictions, use the --predict flag with the path to your test data.
```bash
python main.py --predict --test_data=data/test.csv
```

This command:

1. Loads user_ensemble_models.pkl with previously trained models.
2. Prepares the test data.
3. Generates predictions for each user based on ensemble weights.
4. Saves predictions to ensemble_predictions.csv.

# Dataset Requirements
1. train.csv: Used for training the models.
2. test.csv: Used for generating predictions.

Both files should include:

1. user: Identifier for each unique user.
2. time: Date or timestamp of each record.
3. bought: Target variable (1 if user bought, 0 if not).

Additional time-based features (day_of_week, month, weekday)
will be created during data preparation.

# Project Details

Models Used
1. XGBoost: Extreme Gradient Boosting, tuned for learning rate, subsampling, and other parameters.
2. Random Forest: Ensemble of decision trees with tuning for tree depth and splits.
3. CatBoost: Gradient boosting library optimized for categorical features.
4. LightGBM: Light Gradient Boosting Machine, known for speed and efficiency.

Each model’s parameters are optimized through randomized search using RandomizedSearchCV.

Ensemble Logic

1. Each user’s model is tuned separately, and the best F1-score for each model is recorded.
2. F1-scores are used to weight each model’s contribution when making predictions.
3. Final predictions are based on a weighted majority vote across models.
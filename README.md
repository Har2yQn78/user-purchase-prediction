# Customer Purchase Prediction

## Overview
This project aims to predict customer purchases based on historical data using machine learning models, specifically XGBoost.

## Project Structure
- `data/`: Contains the datasets used for training and testing.
- `notebooks/`: (Optional) Jupyter notebooks for exploratory data analysis.
- `src/`: Contains the source code for data preprocessing, model training, and predictions.
- `requirements.txt`: List of dependencies needed to run the project.

## Installation
1. Clone the repository:
   ```bash
   git clone <https://github.com/Har2yQn78/user-purchase-prediction.git>
   cd user_purchase_prediction

Install the required packages:
```
pip install -r requirements.txt
```
# Usage
Data Preprocessing
Preprocess the train data to extract time-based features.
```
python data_preprocessing.py
```
Feature Engineering
Prepare the data for training by creating dummy variables and separating features from the target.
```
python feature_engineering.py
```
Model Training
Train the XGBoost model for each user and save the best models.
```
python model_training.py
```
Model Prediction
Use the trained models to make predictions on the test data.
```
python model_prediction.py
```
Running the Complete Workflow
To run the entire workflow from preprocessing to making predictions:
```
python main.py
```


# Output
filtered_enhanced_train.csv: Preprocessed train data with additional time-based features.
user_models.pkl: Serialized file containing the trained models for each user.
predictions_output.csv: Output file with predictions for the test data.
Contributing


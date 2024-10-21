"""
Customer Purchase Prediction Package

This package contains modules for loading data, preprocessing data,
training machine learning models, and making predictions for customer
purchase prediction.

Modules:
- data_loading: Functions to load the data from CSV files.
- data_preprocessing: Functions to preprocess the data and add features.
- model_training: Functions to train machine learning models.
- prediction: Functions to add features to the test data and make predictions using trained models.
"""

from .data_loading import load_data
from .data_preprocessing import preprocess_data, save_data
from .model_training import train_user_models
from .prediction import add_features, load_models, predict

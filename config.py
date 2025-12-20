"""
Configuration settings for the Credit Risk Prediction application
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Model directories
MODELS_DIR = BASE_DIR / 'models'
SAVED_MODELS_DIR = MODELS_DIR / 'saved_models'

# Web application settings
class Config:
    """Flask application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True') == 'True'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))

# Model training settings
class ModelConfig:
    """ML model configuration"""
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Model hyperparameters (to be tuned)
    LOGISTIC_REGRESSION_PARAMS = {
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    }
    
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    XGBOOST_PARAMS = {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss'
    }

# Explainability settings
class ExplainabilityConfig:
    """SHAP/LIME configuration"""
    SHAP_SAMPLES = 100  # Number of samples for SHAP computation
    LIME_SAMPLES = 5000  # Number of samples for LIME
    TOP_FEATURES = 10  # Number of top features to display

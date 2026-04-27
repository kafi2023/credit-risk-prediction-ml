"""
Configuration settings for the Credit Risk Prediction application
"""
import os
from pathlib import Path


def _get_env(name: str, default: str) -> str:
    """Return an environment variable or the provided default.

    Treat empty strings as missing so deployment platforms that inject
    blank values do not break module import time configuration.
    """
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _get_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a safe fallback."""
    value = _get_env(name, "true" if default else "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    """Parse an integer environment variable with a safe fallback."""
    raw_value = _get_env(name, str(default)).strip()
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default

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
    SECRET_KEY = _get_env('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = _get_bool_env('FLASK_DEBUG', True)
    HOST = _get_env('FLASK_HOST', '0.0.0.0')
    PORT = _get_int_env('FLASK_PORT', 5000)

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

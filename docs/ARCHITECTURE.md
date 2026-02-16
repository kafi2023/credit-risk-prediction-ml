# System Architecture

## Overview

The Credit Risk Prediction system follows a modular architecture with clear separation of concerns between data processing, model training, prediction, and web interface components.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                     (Flask Web App)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├── HTML/CSS/JS Frontend
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Flask Backend API                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Prediction  │  │Explainability│  │    Models    │      │
│  │   Service    │  │   Service    │  │   Manager    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   ML Pipeline                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     Data     │  │    Model     │  │  Evaluation  │      │
│  │ Preprocessing│─▶│   Training   │─▶│      &       │      │
│  │              │  │              │  │  Selection   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Raw Data   │  │  Processed   │  │    Trained   │      │
│  │   Storage    │  │     Data     │  │    Models    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Layer
- **Raw Data Storage**: Original datasets (CSV, Excel)
- **Processed Data**: Cleaned and transformed data ready for training
- **Model Storage**: Serialized trained models

### 2. ML Pipeline
- **Preprocessing**: Data cleaning, feature engineering, encoding
- **Training**: Model training with cross-validation
- **Evaluation**: Performance metrics and model selection

### 3. Backend API (Flask)
- **Prediction Service**: Handles prediction requests
- **Explainability Service**: Generates SHAP/LIME explanations
- **Model Manager**: Loads and manages trained models

### 4. Frontend (Web Interface)
- **Input Form**: Collect applicant information
- **Results Display**: Show predictions and risk scores
- **Visualization**: Display feature importance and explanations

## Data Flow

1. **Training Phase**:
   - Load raw data → Preprocess → Train models → Evaluate → Save best model

2. **Prediction Phase**:
   - User input → Preprocess → Load model → Predict → Generate explanation → Display

## Technology Stack

- **Backend**: Python, Flask
- **ML**: scikit-learn, XGBoost
- **Explainability**: SHAP, LIME
- **Frontend**: HTML, CSS, JavaScript
- **Data**: pandas, numpy

## Implemented Modules

| Module | File | Description |
|--------|------|-------------|
| Data Loader | `src/preprocessing/data_loader.py` | Downloads UCI German Credit, decodes categories |
| Preprocessor | `src/preprocessing/preprocessor.py` | ColumnTransformer (StandardScaler + OneHotEncoder), SMOTE |
| Training | `src/training/train_models.py` | LR, RF, XGBoost factory + train_all_models |
| Evaluation | `src/training/evaluate.py` | Metrics, CV, comparison tables |
| SHAP | `src/explainability/shap_explainer.py` | TreeSHAP, LinearSHAP, KernelSHAP fallback |
| LIME | `src/explainability/lime_explainer.py` | LimeTabularExplainer wrapper |
| Predictor | `src/prediction/predictor.py` | End-to-end: validate → transform → predict → explain |
| Validator | `src/prediction/input_validator.py` | Range & category validation, schema generation |
| Flask API | `web/app.py` | /predict, /schema, /models, /health endpoints |

## Future Enhancements (Milestone 3+)

- Interactive web UI with SHAP visualisation
- Database integration for storing predictions
- Model versioning and A/B testing
- Deployment on cloud platform

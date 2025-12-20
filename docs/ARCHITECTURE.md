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

## Future Enhancements (Post-Milestone 1)

- Database integration for storing predictions
- User authentication and authorization
- Model versioning and A/B testing
- API rate limiting and caching
- Deployment on cloud platform

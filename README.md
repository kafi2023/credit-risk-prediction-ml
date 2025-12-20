# Credit Risk Prediction with Explainable AI

**BSc Thesis Project - Eötvös Loránd University**  
**Student:** Kafi MD Abdullah Hel (N06WMD)  
**Supervisor:** Md. Easin Arafat

## Overview

An interactive machine learning web application for credit risk prediction with explainable AI capabilities. This tool helps financial institutions make transparent, interpretable lending decisions by combining accurate ML models with visual explanations of prediction factors.

## Problem Statement

Financial institutions struggle to balance accuracy and transparency in credit risk assessment. While modern ML models are highly accurate, they often function as "black boxes." This project bridges that gap by providing both accurate predictions and clear explanations of decision factors.

## Features

- **Multiple ML Algorithms**: Logistic Regression, Random Forest, XGBoost
- **Explainable AI**: SHAP/LIME frameworks for model interpretability
- **Web Interface**: User-friendly Flask application
- **Real-time Predictions**: Instant credit risk assessment
- **Visual Explanations**: Clear feature importance visualization

## Project Structure

```
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Preprocessed data ready for training
├── models/
│   └── saved_models/     # Trained ML models
├── src/
│   ├── preprocessing/    # Data cleaning and feature engineering
│   ├── training/         # Model training scripts
│   ├── prediction/       # Prediction logic
│   └── explainability/   # SHAP/LIME implementation
├── web/
│   ├── static/           # CSS, JavaScript, images
│   ├── templates/        # HTML templates
│   └── app.py            # Flask application
├── tests/                # Unit and integration tests
├── docs/                 # Documentation
├── notebooks/            # Jupyter notebooks for exploration
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kafi2023/credit-risk-prediction-ml.git
cd credit-risk-prediction-ml
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

```bash
python src/training/train_models.py
```

### Running the Web Application

```bash
python web/app.py
```

Access the application at `http://localhost:5000`

## Milestones

- **Milestone 1** (Dec 20, 2025): Initial setup, research, and project skeleton ✓
- **Milestone 2** (Feb 20, 2026): Core ML implementation and backend
- **Milestone 3** (Mar 25, 2026): UI development and optimization
- **Milestone 4** (Apr 15, 2026): Finalization and documentation

## Technologies

- **Backend**: Python, Flask
- **ML Libraries**: scikit-learn, XGBoost
- **Explainability**: SHAP, LIME
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, plotly

## License

This project is developed as part of academic research at Eötvös Loránd University.

## Contact

- **Author**: Kafi MD Abdullah Hel
- **Email**: [Your Email]
- **Repository**: https://github.com/kafi2023/credit-risk-prediction-ml

# Development Guide

## Setup Development Environment

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/kafi2023/credit-risk-prediction-ml.git
cd credit-risk-prediction-ml
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Project Structure

```
credit-risk-prediction-ml/
├── data/                   # Data storage
│   ├── raw/               # Original datasets
│   └── processed/         # Preprocessed data
├── models/                # Trained models
│   └── saved_models/      # Serialized models
├── src/                   # Source code
│   ├── preprocessing/     # Data preprocessing
│   ├── training/          # Model training
│   ├── prediction/        # Prediction logic
│   └── explainability/    # SHAP/LIME
├── web/                   # Flask web app
│   ├── static/           # CSS, JS, images
│   ├── templates/        # HTML templates
│   └── app.py            # Main Flask app
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── notebooks/             # Jupyter notebooks
```

## Running the Application

### Development Mode

Start the Flask development server:
```bash
python web/app.py
```

Access at: http://localhost:5000

### Running Tests

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

## Development Workflow

### 1. Feature Development
- Create a new branch: `git checkout -b feature/your-feature`
- Make changes
- Write tests
- Commit: `git commit -m "Add feature"`
- Push: `git push origin feature/your-feature`
- Create pull request

### 2. Code Standards
- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions small and focused

### 3. Testing
- Write unit tests for new functionality
- Maintain test coverage above 80%
- Test edge cases and error handling

## Milestones

### Milestone 1 (Dec 20, 2025) — ✅ COMPLETED
- [x] Set up repository
- [x] Create project skeleton
- [x] Research datasets (→ DATASET_RESEARCH.md)
- [x] Research algorithms (→ ALGORITHM_RESEARCH.md)
- [x] Research explainability (→ EXPLAINABILITY_RESEARCH.md)
- [x] Create architecture diagram (→ ARCHITECTURE.md)

### Milestone 2 (Feb 20, 2026) — ✅ COMPLETED
- [x] Download & load German Credit Dataset (data_loader.py)
- [x] Exploratory Data Analysis notebook (01_eda.ipynb)
- [x] Preprocessing pipeline with ColumnTransformer (preprocessor.py)
- [x] Train LR, RF, XGBoost models (train_models.py)
- [x] Evaluation module with metrics & CV (evaluate.py)
- [x] SHAP explainability module (shap_explainer.py)
- [x] LIME explainability module (lime_explainer.py)
- [x] Prediction module with input validation (predictor.py, input_validator.py)
- [x] Flask backend integration (/predict, /schema, /models)
- [x] Unit tests — 103 tests, 84% coverage

### Milestone 3 (Mar 25, 2026)
- [ ] Build interactive UI with form & visualisations
- [ ] Improve model performance (hyperparameter tuning)
- [ ] Integration tests
- [ ] Documentation improvements

### Milestone 4 (Apr 15, 2026)
- [ ] Final testing
- [ ] Complete documentation
- [ ] Thesis draft

## Troubleshooting

### Common Issues

**Issue**: Import errors
**Solution**: Ensure virtual environment is activated and dependencies installed

**Issue**: Flask app won't start
**Solution**: Check if port 5000 is available, try a different port

**Issue**: Model training fails
**Solution**: Check data format and preprocessing steps

## Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

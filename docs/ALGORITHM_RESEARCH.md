# Machine Learning Algorithms for Credit Risk - Research Report

**Date:** December 20, 2025  
**Project:** Credit Risk Prediction with Explainable AI  
**Milestone:** 1 - Initial Setup & Planning

## Overview

This document analyzes machine learning algorithms suitable for credit risk prediction. The focus is on algorithms that balance predictive accuracy with interpretability, supporting our explainable AI objectives.

---

## Selected Algorithms

### 1. Logistic Regression

#### Description
Linear model for binary classification using the logistic (sigmoid) function to predict probabilities.

#### Mathematical Foundation
- **Function:** P(Y=1|X) = 1 / (1 + e^(-(β₀ + β₁X₁ + ... + βₙXₙ)))
- **Loss:** Log-loss (binary cross-entropy)
- **Optimization:** Gradient descent, L-BFGS

#### Strengths
- ✅ **Highly interpretable** - Coefficient weights show feature importance
- ✅ **Fast training** - Efficient even on large datasets
- ✅ **Probabilistic output** - Natural risk scores (0-1)
- ✅ **Well-understood** - Extensive theoretical foundation
- ✅ **Baseline model** - Standard for comparison
- ✅ **Regularization** - L1/L2 to prevent overfitting

#### Weaknesses
- ❌ **Linear relationships** - Cannot capture complex non-linear patterns
- ❌ **Feature engineering** - Requires manual interaction terms
- ❌ **Assumes independence** - May not model feature interactions well

#### Credit Risk Suitability
**Rating: ⭐⭐⭐⭐**

**Use Case:**
- Baseline model for comparison
- When interpretability is critical
- Regulatory compliance requirements
- Understanding feature contributions

#### Implementation Details
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',  # Handle imbalance
    penalty='l2',             # Regularization
    C=1.0                     # Inverse regularization strength
)
```

#### Expected Performance
- **Accuracy:** 70-75% (German Credit Data)
- **AUC-ROC:** 0.72-0.78
- **Training Time:** < 1 second
- **Prediction Time:** Milliseconds

---

### 2. Random Forest

#### Description
Ensemble of decision trees using bagging and random feature selection to reduce overfitting and improve generalization.

#### Mathematical Foundation
- **Ensemble:** Majority voting from multiple trees
- **Bagging:** Bootstrap aggregating for variance reduction
- **Random Features:** Subset of features per split

#### Strengths
- ✅ **High accuracy** - Often outperforms single models
- ✅ **Feature importance** - Built-in importance metrics
- ✅ **Non-linear** - Captures complex patterns
- ✅ **Handles interactions** - Automatically models feature combinations
- ✅ **Robust to outliers** - Tree-based resilience
- ✅ **No scaling needed** - Works with raw features
- ✅ **Class imbalance** - Can use balanced class weights

#### Weaknesses
- ❌ **Black box** - Individual predictions less interpretable
- ❌ **Slower training** - Multiple trees to build
- ❌ **Memory intensive** - Stores multiple trees
- ❌ **Overfitting risk** - Without proper tuning

#### Credit Risk Suitability
**Rating: ⭐⭐⭐⭐⭐**

**Use Case:**
- Primary prediction model
- When accuracy is priority
- Complex feature interactions expected
- Balanced interpretability needs (via SHAP/LIME)

#### Implementation Details
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,         # Number of trees
    max_depth=10,             # Prevent overfitting
    min_samples_split=20,     # Minimum samples to split
    min_samples_leaf=10,      # Minimum samples in leaf
    max_features='sqrt',      # Random feature subset
    random_state=42,
    n_jobs=-1,                # Parallel processing
    class_weight='balanced'   # Handle imbalance
)
```

#### Expected Performance
- **Accuracy:** 75-80% (German Credit Data)
- **AUC-ROC:** 0.78-0.85
- **Training Time:** 2-5 seconds
- **Prediction Time:** ~10ms

#### Hyperparameter Tuning
Key parameters to optimize:
- `n_estimators`: 50, 100, 200, 500
- `max_depth`: 5, 10, 15, 20, None
- `min_samples_split`: 10, 20, 50
- `max_features`: 'sqrt', 'log2', None

---

### 3. XGBoost (Extreme Gradient Boosting)

#### Description
Advanced gradient boosting framework using sequential tree building with regularization and optimization techniques.

#### Mathematical Foundation
- **Boosting:** Sequential learning, correcting previous errors
- **Gradient:** Optimizes loss function using gradients
- **Regularization:** L1/L2 penalties on tree complexity
- **Objective:** Loss + Ω(f) where Ω penalizes complexity

#### Strengths
- ✅ **State-of-the-art accuracy** - Often wins ML competitions
- ✅ **Feature importance** - Gain, cover, frequency metrics
- ✅ **Built-in regularization** - Reduces overfitting
- ✅ **Handles missing data** - Native support
- ✅ **Fast training** - Optimized C++ implementation
- ✅ **Cross-validation** - Built-in CV support
- ✅ **Custom objectives** - Flexible loss functions
- ✅ **Class imbalance** - scale_pos_weight parameter

#### Weaknesses
- ❌ **Complex tuning** - Many hyperparameters
- ❌ **Less interpretable** - Boosting adds complexity
- ❌ **Requires scaling** - Better with normalized features
- ❌ **Overfitting risk** - Without careful regularization

#### Credit Risk Suitability
**Rating: ⭐⭐⭐⭐⭐**

**Use Case:**
- Best-performing model
- Competition-grade accuracy
- Complex credit patterns
- When combined with SHAP for interpretability

#### Implementation Details
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,            # Row sampling
    colsample_bytree=0.8,     # Column sampling
    gamma=0,                  # Min loss reduction to split
    reg_alpha=0,              # L1 regularization
    reg_lambda=1,             # L2 regularization
    scale_pos_weight=1,       # Handle imbalance
    random_state=42,
    eval_metric='logloss'
)
```

#### Expected Performance
- **Accuracy:** 78-83% (German Credit Data)
- **AUC-ROC:** 0.82-0.88
- **Training Time:** 1-3 seconds
- **Prediction Time:** ~5ms

#### Hyperparameter Tuning
Key parameters to optimize:
- `n_estimators`: 50, 100, 200, 500
- `max_depth`: 3, 5, 7, 9
- `learning_rate`: 0.01, 0.05, 0.1, 0.3
- `subsample`: 0.6, 0.8, 1.0
- `colsample_bytree`: 0.6, 0.8, 1.0
- `gamma`: 0, 0.1, 0.5, 1
- `reg_alpha`: 0, 0.1, 1
- `reg_lambda`: 1, 10, 100

---

## Algorithm Comparison

### Performance Comparison

| Algorithm | Accuracy | AUC-ROC | Speed | Interpretability | Complexity |
|-----------|----------|---------|-------|------------------|------------|
| Logistic Regression | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| XGBoost | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

### Use Cases

| Scenario | Recommended Algorithm | Reason |
|----------|----------------------|---------|
| Regulatory compliance | Logistic Regression | High interpretability |
| Maximum accuracy | XGBoost | Best performance |
| Balanced approach | Random Forest | Good accuracy + interpretability |
| Fast deployment | Logistic Regression | Fastest training/prediction |
| Complex patterns | XGBoost | Handles non-linearity best |
| Limited data | Logistic Regression | Less prone to overfit |

---

## Model Evaluation Strategy

### Metrics to Use

1. **Accuracy** - Overall correct predictions
   - Standard metric but can be misleading with imbalanced data

2. **Precision** - TP / (TP + FP)
   - Important: Avoid false positives (rejecting good applicants)

3. **Recall** - TP / (TP + FN)
   - Important: Catch actual defaults (minimize bad loans)

4. **F1-Score** - Harmonic mean of precision and recall
   - Balances both concerns

5. **AUC-ROC** - Area Under ROC Curve
   - Overall model discrimination ability
   - Best single metric for credit risk

6. **AUC-PR** - Precision-Recall AUC
   - Better for imbalanced datasets

7. **Confusion Matrix** - Detailed breakdown
   - Shows Type I and Type II errors

8. **Cost-Sensitive Metrics**
   - False Negative Cost: Approving bad loan (higher cost)
   - False Positive Cost: Rejecting good applicant (lower cost)

### Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

- Use stratified K-fold to maintain class distribution
- 5-fold cross-validation for robust evaluation
- Report mean and standard deviation of metrics

---

## Implementation Workflow

### Phase 1: Baseline (Logistic Regression)
1. Train simple logistic regression
2. Establish baseline metrics
3. Analyze coefficients for feature insights
4. Document interpretation

### Phase 2: Ensemble Learning (Random Forest)
1. Train Random Forest with default parameters
2. Hyperparameter tuning using Grid/Random Search
3. Compare with baseline
4. Extract feature importance

### Phase 3: Advanced Model (XGBoost)
1. Train XGBoost with default parameters
2. Extensive hyperparameter tuning
3. Early stopping to prevent overfitting
4. Compare all three models

### Phase 4: Model Selection
1. Compare metrics across all models
2. Consider accuracy vs. interpretability tradeoff
3. Select best model for deployment
4. Integrate with SHAP/LIME for explainability

---

## Handling Class Imbalance

### Techniques to Apply

1. **Class Weights**
   ```python
   class_weight='balanced'  # Automatically adjust weights
   ```

2. **SMOTE** (Synthetic Minority Over-sampling)
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   ```

3. **Threshold Adjustment**
   - Adjust decision threshold based on cost-benefit analysis
   - Default is 0.5, but can optimize for business objectives

4. **Ensemble Methods**
   - Random Forest and XGBoost handle imbalance better naturally

---

## Feature Engineering Considerations

### For All Models:
- **Scaling:** StandardScaler or MinMaxScaler (except tree-based)
- **Encoding:** One-hot encoding for categorical features
- **Missing Values:** Imputation strategies
- **Outliers:** Detection and handling

### Advanced Features:
- **Interaction Terms:** For Logistic Regression
- **Polynomial Features:** Non-linear relationships
- **Domain-Specific:** Debt-to-income ratios, credit utilization

---

## Expected Timeline

| Phase | Task | Duration | Milestone |
|-------|------|----------|-----------|
| 1 | Logistic Regression baseline | 1 week | Milestone 2 |
| 2 | Random Forest implementation | 1 week | Milestone 2 |
| 3 | XGBoost implementation | 1 week | Milestone 2 |
| 4 | Hyperparameter tuning | 1 week | Milestone 2 |
| 5 | Model comparison & selection | 2 days | Milestone 2 |

---

## Conclusion

This project will implement and compare **three complementary algorithms**:

1. **Logistic Regression** - Interpretable baseline
2. **Random Forest** - Balanced accuracy and interpretability
3. **XGBoost** - Maximum accuracy

The combination allows us to:
- Establish a simple, interpretable baseline
- Achieve competitive accuracy with ensemble methods
- Demonstrate explainability across different model complexities
- Provide comprehensive comparison for thesis

The final model selection will balance accuracy requirements with explainability needs, supporting the thesis objective of transparent credit risk prediction.

---

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
3. Breiman, L. (2001). Random Forests. Machine Learning
4. scikit-learn documentation: https://scikit-learn.org/
5. XGBoost documentation: https://xgboost.readthedocs.io/

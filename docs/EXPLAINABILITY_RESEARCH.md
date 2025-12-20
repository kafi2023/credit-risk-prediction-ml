# Explainability Frameworks (SHAP & LIME) - Research Report

**Date:** December 20, 2025  
**Project:** Credit Risk Prediction with Explainable AI  
**Milestone:** 1 - Initial Setup & Planning

## Overview

This document analyzes explainability frameworks for making machine learning models interpretable. As credit risk decisions significantly impact individuals' lives and are subject to regulatory scrutiny, explaining model predictions is crucial. We compare SHAP and LIME, the two leading model-agnostic explanation frameworks.

---

## Why Explainability Matters in Credit Risk

### Business Reasons
- **Trust:** Applicants deserve to understand why they were approved/rejected
- **Improvement:** Applicants can take actions to improve creditworthiness
- **Validation:** Loan officers can verify model decisions
- **Debugging:** Identify when models make unreasonable predictions

### Regulatory Reasons
- **GDPR (EU):** Right to explanation for automated decisions
- **Fair Lending Laws:** Equal Credit Opportunity Act (ECOA)
- **Model Risk Management:** Basel III compliance
- **Adverse Action Notices:** Required explanation for credit denial

### Ethical Reasons
- **Fairness:** Detect and mitigate bias (gender, race, age)
- **Accountability:** Responsible AI principles
- **Transparency:** Black-box models undermine trust

---

## SHAP (SHapley Additive exPlanations)

### Overview

SHAP is based on game theory (Shapley values) and provides a unified framework for interpreting model predictions by calculating each feature's contribution to the prediction.

### Theoretical Foundation

**Shapley Values (Game Theory)**
- Originally used to fairly distribute payouts among players in cooperative games
- Each feature is a "player" contributing to the final "payout" (prediction)
- Calculates average marginal contribution across all possible feature combinations

**Mathematical Definition:**

φᵢ = Σ [|S|! × (|F| - |S| - 1)!] / |F|! × [f(S ∪ {i}) - f(S)]

Where:
- φᵢ = Shapley value for feature i
- S = subset of features
- F = all features
- f(S) = model prediction with feature subset S

### Key Properties

1. **Local Accuracy:** Explanation model matches original model locally
2. **Missingness:** Missing features have zero impact
3. **Consistency:** If a feature's contribution increases, Shapley value doesn't decrease
4. **Additivity:** For ensemble models, Shapley values add up

### SHAP Variants

#### 1. TreeSHAP
- **For:** Tree-based models (Random Forest, XGBoost)
- **Speed:** Fast (polynomial time)
- **Accuracy:** Exact Shapley values
- **Best for:** Our project (we use Random Forest & XGBoost)

#### 2. KernelSHAP
- **For:** Any model (model-agnostic)
- **Speed:** Slower (approximation)
- **Accuracy:** Approximate Shapley values
- **Best for:** Complex models without tree structure

#### 3. DeepSHAP
- **For:** Deep neural networks
- **Speed:** Fast
- **Not relevant:** We're not using deep learning

#### 4. LinearSHAP
- **For:** Linear models
- **Speed:** Very fast
- **Best for:** Logistic Regression baseline

### Advantages

✅ **Theoretically Sound**
- Based on game theory with solid mathematical foundation
- Unique solution satisfying desirable properties

✅ **Consistent**
- Same interpretation across different models
- Fair attribution of feature importance

✅ **Global and Local**
- Explain individual predictions (local)
- Aggregate for overall feature importance (global)

✅ **Feature Interactions**
- Can capture interaction effects between features
- Shows how features work together

✅ **Fast for Trees**
- TreeSHAP is highly optimized for tree-based models
- Practical for real-time explanations

✅ **Comprehensive Visualizations**
- Force plots, waterfall plots, beeswarm plots
- Summary plots for global importance
- Dependence plots for feature relationships

### Disadvantages

❌ **Computational Cost**
- KernelSHAP can be slow for complex models
- Requires many model evaluations

❌ **Complexity**
- Harder to understand than simpler methods
- Requires explanation of Shapley values to stakeholders

❌ **Correlation Issues**
- Can behave unexpectedly with highly correlated features

### Implementation

```python
import shap

# For tree-based models (Fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# For any model (Slower)
explainer = shap.KernelExplainer(model.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)

# Visualizations
shap.summary_plot(shap_values, X)  # Global importance
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])  # Individual
shap.waterfall_plot(shap_values[0])  # Individual breakdown
```

### Visualization Types

1. **Force Plot:** Shows how features push prediction from base value
2. **Waterfall Plot:** Step-by-step feature contribution
3. **Summary Plot:** Global feature importance with distribution
4. **Dependence Plot:** Relationship between feature and SHAP values
5. **Decision Plot:** Path from base value to final prediction

### Credit Risk Suitability
**Rating: ⭐⭐⭐⭐⭐**

**Perfect for:**
- Tree-based models (Random Forest, XGBoost)
- When theoretical soundness matters
- Regulatory compliance
- Academic/thesis work

---

## LIME (Local Interpretable Model-agnostic Explanations)

### Overview

LIME explains individual predictions by approximating the complex model locally with a simple, interpretable model (like linear regression).

### Theoretical Foundation

**Core Idea:**
- Complex models are hard to interpret globally
- But they might be approximated locally (near a single prediction)
- Train a simple model on perturbed samples around the instance
- Use the simple model's weights as explanations

**Mathematical Definition:**

ξ(x) = argmin L(f, g, πₓ) + Ω(g)

Where:
- ξ(x) = explanation for instance x
- f = complex model
- g = simple interpretable model
- πₓ = proximity measure to x
- L = loss function (how well g approximates f)
- Ω = complexity penalty for g

### How LIME Works

1. **Perturb:** Generate new samples by perturbing features of the instance
2. **Predict:** Get complex model's predictions for perturbed samples
3. **Weight:** Weight samples by proximity to original instance
4. **Train:** Train simple model (e.g., linear regression) on weighted samples
5. **Explain:** Use simple model's coefficients as feature importance

### Advantages

✅ **Model-Agnostic**
- Works with any black-box model
- No access to model internals needed

✅ **Intuitive**
- Easy to understand (linear coefficients)
- Familiar to non-technical stakeholders

✅ **Local Fidelity**
- Accurately represents model behavior locally
- Good for explaining individual predictions

✅ **Flexible**
- Can explain any prediction task
- Works with text, images, tabular data

✅ **Implementation Simplicity**
- Straightforward to implement
- Few hyperparameters

### Disadvantages

❌ **Instability**
- Small changes in input can lead to different explanations
- Randomness in sampling process

❌ **Local Only**
- Only explains individual predictions
- No global interpretation (must aggregate manually)

❌ **Sampling Dependency**
- Quality depends on sampling strategy
- May miss important regions

❌ **Computational Cost**
- Requires training a model for each explanation
- Can be slow for many predictions

❌ **Hyperparameter Sensitivity**
- Number of samples affects quality
- Kernel width affects locality

❌ **No Theoretical Guarantees**
- Lacks formal properties like SHAP
- Can give inconsistent explanations

### Implementation

```python
from lime import lime_tabular

# Create explainer
explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Good', 'Bad'],
    mode='classification',
    discretize_continuous=True
)

# Explain instance
explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10,      # Top 10 features
    num_samples=5000      # Samples for local model
)

# Visualize
explanation.show_in_notebook()
explanation.as_pyplot_figure()
```

### Hyperparameters

- **num_features:** How many features to show (default: all)
- **num_samples:** Perturbations for local model (default: 5000)
- **kernel_width:** Proximity measure (default: auto)
- **discretize_continuous:** Bin continuous features (True/False)

### Visualization Types

1. **Feature Importance Plot:** Bar chart of top features
2. **Table View:** Feature values and contributions
3. **HTML Report:** Interactive explanation

### Credit Risk Suitability
**Rating: ⭐⭐⭐⭐**

**Good for:**
- Any model type
- Quick prototyping
- Simple explanations for stakeholders
- When SHAP is too slow

---

## SHAP vs. LIME Comparison

### Head-to-Head Comparison

| Aspect | SHAP | LIME |
|--------|------|------|
| **Theoretical Foundation** | Game theory (Shapley values) | Local linear approximation |
| **Guarantees** | Unique, consistent solution | No formal guarantees |
| **Scope** | Local + Global | Local only |
| **Speed (Trees)** | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐ Moderate |
| **Speed (Other)** | ⭐⭐ Slow | ⭐⭐⭐ Moderate |
| **Stability** | ⭐⭐⭐⭐ High | ⭐⭐ Low |
| **Interpretability** | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ High |
| **Model-Agnostic** | ✅ Yes | ✅ Yes |
| **Consistency** | ⭐⭐⭐⭐⭐ High | ⭐⭐ Low |
| **Visualization** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good |
| **Learning Curve** | Steep | Gentle |
| **Academic Credibility** | ⭐⭐⭐⭐⭐ High | ⭐⭐⭐⭐ Good |

### When to Use Each

#### Use SHAP When:
- Using tree-based models (Random Forest, XGBoost) ✅
- Need theoretical rigor for thesis/publication ✅
- Regulatory compliance requires consistent explanations ✅
- Want both local and global interpretability ✅
- Concerned about explanation stability ✅

#### Use LIME When:
- Need quick prototyping
- Model is not tree-based and SHAP is too slow
- Stakeholders prefer simpler linear explanations
- Only need local explanations
- Want maximum simplicity

---

## Recommendation for This Project

### Primary Framework: **SHAP** ⭐⭐⭐⭐⭐

**Rationale:**

1. **Tree-Based Models:** We're using Random Forest and XGBoost
   - TreeSHAP is extremely fast and provides exact values
   - Perfect fit for our models

2. **Thesis Requirements:** 
   - Strong theoretical foundation
   - Academic credibility
   - Comprehensive documentation

3. **Explainability Goals:**
   - Both local (individual applicants) and global (overall) insights
   - Consistent explanations across predictions
   - Rich visualizations for thesis

4. **Regulatory Context:**
   - Shapley values are mathematically sound
   - Fair attribution aligns with compliance needs

### Secondary Framework: **LIME** ⭐⭐⭐⭐

**Use Cases:**

1. **Comparison:** Show different explanation approaches in thesis
2. **Validation:** Compare SHAP and LIME explanations for consistency
3. **Simplicity:** When presenting to non-technical audience
4. **Baseline:** Logistic Regression explanations (though SHAP works too)

---

## Implementation Plan

### Milestone 2: SHAP Integration

**Week 1-2:**
1. Install SHAP library
2. Implement TreeSHAP for Random Forest
3. Implement TreeSHAP for XGBoost
4. Generate force plots for individual predictions
5. Create summary plots for global importance

**Week 3:**
6. Implement dependence plots
7. Analyze feature interactions
8. Document findings

### Milestone 3: LIME Integration & Comparison

**Week 1:**
1. Install LIME library
2. Implement LIME for all models
3. Compare LIME vs. SHAP explanations

**Week 2:**
4. Validate consistency
5. Document differences
6. Choose primary framework for deployment

### Milestone 3: Web Integration

**Week 3:**
7. Integrate SHAP into Flask app
8. Generate explanations in real-time
9. Create visualizations for web display
10. Export explanations as images/HTML

---

## Visualization Strategy

### For Web Application:

1. **Individual Prediction Page:**
   - SHAP force plot (horizontal bar showing contribution)
   - Top 10 features with contributions
   - Risk score with explanation

2. **Feature Importance Page:**
   - SHAP summary plot (global importance)
   - Feature distributions with SHAP values
   - Interaction effects

3. **Comparison Page:**
   - Side-by-side SHAP vs. LIME
   - Consistency analysis

### For Thesis:

1. **Introduction:** Why explainability matters
2. **Methodology:** SHAP theory and implementation
3. **Results:** Visualizations with analysis
4. **Comparison:** SHAP vs. LIME empirical comparison
5. **Discussion:** Insights gained from explanations

---

## Code Examples

### Complete SHAP Workflow

```python
import shap
import matplotlib.pyplot as plt

# Train model (XGBoost example)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Individual explanation
i = 0  # First instance
shap.force_plot(
    explainer.expected_value,
    shap_values[i],
    X_test.iloc[i],
    matplotlib=True
)

# Global importance
shap.summary_plot(shap_values, X_test)

# Feature dependence
shap.dependence_plot('credit_amount', shap_values, X_test)

# Waterfall for single prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[i],
    base_values=explainer.expected_value,
    data=X_test.iloc[i],
    feature_names=X_test.columns
))
```

### Complete LIME Workflow

```python
from lime import lime_tabular

# Create explainer
explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Good Credit', 'Bad Credit'],
    mode='classification'
)

# Explain prediction
i = 0
exp = explainer.explain_instance(
    X_test.iloc[i].values,
    model.predict_proba,
    num_features=10
)

# Show explanation
exp.show_in_notebook(show_table=True)
exp.as_pyplot_figure()

# Save to file
exp.save_to_file('explanation.html')
```

---

## Performance Considerations

### SHAP Performance

| Model | Method | Time per Instance | Scalability |
|-------|--------|------------------|-------------|
| Logistic Regression | LinearSHAP | ~1ms | Excellent |
| Random Forest | TreeSHAP | ~10ms | Excellent |
| XGBoost | TreeSHAP | ~5ms | Excellent |
| Any Model | KernelSHAP | ~500ms | Poor |

### LIME Performance

| Configuration | Time per Instance | Quality |
|--------------|------------------|---------|
| 1000 samples | ~100ms | Low |
| 5000 samples | ~500ms | Good |
| 10000 samples | ~1000ms | Better |

### Optimization Tips

1. **SHAP:**
   - Use TreeSHAP for tree models (not KernelSHAP)
   - Pre-compute for common scenarios
   - Cache explanations
   - Use check_additivity=False for speed

2. **LIME:**
   - Reduce num_samples for faster results
   - Use discretize_continuous for simpler explanations
   - Parallel processing for batch explanations

---

## Conclusion

For this credit risk prediction project:

1. **Primary:** SHAP (TreeSHAP specifically)
   - Fast and accurate for our tree-based models
   - Theoretically sound for thesis
   - Excellent visualizations
   - Both local and global insights

2. **Secondary:** LIME
   - For comparison and validation
   - Simpler stakeholder communication
   - Complementary perspective

This combination provides:
- ✅ Robust explainability
- ✅ Academic rigor
- ✅ Practical usability
- ✅ Comprehensive thesis content
- ✅ Regulatory compliance

The integration of both frameworks strengthens the thesis by demonstrating thorough analysis of explainability approaches while relying primarily on the theoretically superior SHAP for production use.

---

## References

1. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP)
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)
3. Molnar, C. (2022). Interpretable Machine Learning
4. SHAP documentation: https://shap.readthedocs.io/
5. LIME documentation: https://lime-ml.readthedocs.io/
6. Shapley, L. S. (1953). A value for n-person games

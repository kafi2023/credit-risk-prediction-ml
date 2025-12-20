# Credit Risk Datasets - Research Report

**Date:** December 20, 2025  
**Project:** Credit Risk Prediction with Explainable AI  
**Milestone:** 1 - Initial Setup & Planning

## Overview

This document provides a comprehensive analysis of publicly available credit risk datasets suitable for our thesis project. The goal is to identify datasets that contain sufficient features, quality data, and real-world applicability for credit risk prediction.

## Recommended Datasets

### 1. German Credit Data (Statlog)

**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

**Description:**
- One of the most widely used datasets for credit risk assessment
- Contains 1,000 instances with 20 attributes (7 numerical, 13 categorical)
- Binary classification: Good vs. Bad credit risk

**Features Include:**
- Account status and duration
- Credit history
- Purpose of credit
- Credit amount
- Employment status
- Personal status and sex
- Property ownership
- Age
- Other installments
- Housing type
- Job type

**Pros:**
- Well-documented and widely studied
- Balanced features (demographic, financial, behavioral)
- Good for benchmarking against existing research
- Real-world data from a German bank

**Cons:**
- Relatively small dataset (1,000 instances)
- Data from 1994 (may not reflect current trends)
- Imbalanced classes (70% good, 30% bad)

**Suitability:** ⭐⭐⭐⭐⭐ Excellent for initial prototyping and thesis work

---

### 2. Default of Credit Card Clients Dataset

**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

**Description:**
- Credit card default data from Taiwan
- 30,000 instances with 24 attributes
- Binary classification: Default vs. Non-default payment

**Features Include:**
- Credit limit
- Gender, education, marital status, age
- Payment history (6 months)
- Bill amounts (6 months)
- Payment amounts (6 months)

**Pros:**
- Larger dataset (30,000 samples)
- Time-series payment history
- Rich feature set
- Real credit card data

**Cons:**
- Focused on credit cards (not general loans)
- May require more preprocessing
- Imbalanced (default rate ~22%)

**Suitability:** ⭐⭐⭐⭐⭐ Excellent for robust model training

---

### 3. Lending Club Loan Data

**Source:** Kaggle  
**URL:** https://www.kaggle.com/datasets/wordsforthewise/lending-club

**Description:**
- Peer-to-peer lending data from Lending Club
- Over 2 million loan records
- Comprehensive feature set (150+ columns)

**Features Include:**
- Loan amount and term
- Interest rate
- Borrower income and employment
- DTI (Debt-to-Income) ratio
- Credit history length
- Number of credit inquiries
- Home ownership status
- Loan purpose
- Loan status (fully paid, charged off, etc.)

**Pros:**
- Very large dataset
- Extremely comprehensive features
- Recent data (2007-2018)
- Real-world P2P lending platform

**Cons:**
- Requires significant preprocessing
- Large file size
- May have missing values
- Complex feature engineering needed

**Suitability:** ⭐⭐⭐⭐ Good for advanced analysis (may be too large for thesis scope)

---

### 4. Give Me Some Credit (Kaggle Competition)

**Source:** Kaggle  
**URL:** https://www.kaggle.com/c/GiveMeSomeCredit

**Description:**
- Credit default prediction dataset
- ~150,000 training instances
- 11 features including delinquency indicators

**Features Include:**
- Revolving utilization of unsecured lines
- Age
- Number of times 30-59, 60-89, 90+ days past due
- Debt ratio
- Monthly income
- Number of open credit lines
- Number of real estate loans
- Number of dependents

**Pros:**
- Good size for machine learning
- Focused on key credit indicators
- Kaggle competition (benchmarks available)
- Clean feature set

**Cons:**
- Some missing values
- Limited demographic features
- Competition dataset (well-studied)

**Suitability:** ⭐⭐⭐⭐ Good for model comparison

---

### 5. Home Credit Default Risk

**Source:** Kaggle  
**URL:** https://www.kaggle.com/c/home-credit-default-risk

**Description:**
- Credit default prediction for unbanked population
- 300,000+ instances
- Multiple related datasets (applications, credit bureau, previous applications)

**Features Include:**
- Comprehensive applicant information
- External data sources
- Previous credit history
- Installment payments

**Pros:**
- Very comprehensive
- Multiple data sources
- Real-world unbanked population focus
- Recent data

**Cons:**
- Complex multi-table structure
- Requires extensive merging/preprocessing
- May be too complex for BSc thesis

**Suitability:** ⭐⭐⭐ Complex but interesting for advanced work

---

## Recommended Choice for This Project

### Primary Dataset: **German Credit Data**

**Rationale:**
1. **Perfect thesis scope** - 1,000 instances is manageable
2. **Well-documented** - Extensive literature for reference
3. **Balanced features** - Good mix of financial and demographic data
4. **Explainability focus** - Simpler feature set makes explanations clearer
5. **Quick prototyping** - Can train models quickly
6. **Benchmarking** - Easy to compare with existing research

### Secondary Dataset: **Default of Credit Card Clients**

**Rationale:**
1. **Validation** - Test model generalization on different dataset
2. **Larger scale** - Demonstrate scalability
3. **Time-series features** - Shows handling of temporal data
4. **Comparison** - Compare performance across datasets

---

## Data Characteristics Comparison

| Dataset | Size | Features | Class Balance | Complexity | Thesis Fit |
|---------|------|----------|---------------|------------|------------|
| German Credit | 1,000 | 20 | 70/30 | Low | ⭐⭐⭐⭐⭐ |
| Credit Card Default | 30,000 | 24 | 78/22 | Medium | ⭐⭐⭐⭐⭐ |
| Lending Club | 2M+ | 150+ | Varies | High | ⭐⭐⭐ |
| Give Me Credit | 150K | 11 | 93/7 | Medium | ⭐⭐⭐⭐ |
| Home Credit | 300K+ | 100+ | 92/8 | Very High | ⭐⭐⭐ |

---

## Implementation Plan

### Phase 1: German Credit Data (Milestone 2)
- Download and explore dataset
- Perform EDA (Exploratory Data Analysis)
- Handle categorical encoding
- Address class imbalance
- Train baseline models

### Phase 2: Credit Card Default (Milestone 3)
- Validate model on larger dataset
- Compare performance metrics
- Test explainability across datasets
- Document findings

---

## Data Preprocessing Requirements

### For German Credit Data:
1. **Categorical Encoding**: One-hot or label encoding for 13 categorical features
2. **Normalization**: Scale numerical features
3. **Class Imbalance**: Consider SMOTE or class weights
4. **Feature Engineering**: Create interaction features if needed

### For Credit Card Default:
1. **Payment History**: Aggregate or use as time-series
2. **Missing Values**: Handle missing demographic data
3. **Scaling**: Normalize bill amounts and payments
4. **Feature Selection**: Reduce dimensionality if needed

---

## Ethical Considerations

- **Bias**: Check for gender, age, and demographic bias
- **Fairness**: Ensure equal treatment across groups
- **Transparency**: Use explainable features
- **Privacy**: All datasets are anonymized and publicly available

---

## References

1. UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/
2. Kaggle Datasets: https://www.kaggle.com/datasets
3. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
4. Various Kaggle competition documentation

---

## Conclusion

The **German Credit Data** is the optimal choice for this thesis project due to its manageable size, comprehensive documentation, and suitability for explainable AI research. The **Credit Card Default dataset** will serve as an excellent secondary validation dataset to demonstrate model generalization.

Both datasets provide sufficient complexity for a BSc thesis while remaining manageable within the project timeline. They offer opportunities to demonstrate data preprocessing, model training, evaluation, and explainability - all key objectives of this project.

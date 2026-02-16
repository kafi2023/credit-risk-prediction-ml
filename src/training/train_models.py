"""
Train Logistic Regression, Random Forest, and XGBoost on the
German Credit dataset.  Evaluate with stratified cross-validation,
compare results, and persist the best models.
"""

import joblib
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocessing.preprocessor import (
    prepare_data,
    load_processed_data,
    MODELS_DIR,
)
from src.training.evaluate import (
    evaluate_model,
    cross_validate_model,
    comparison_table,
    cv_comparison_table,
    print_classification_report,
)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _class_weight_ratio(y) -> float:
    """Compute scale_pos_weight for XGBoost from target array."""
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    return n_neg / n_pos if n_pos > 0 else 1.0


def build_logistic_regression(**kwargs) -> LogisticRegression:
    defaults = dict(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        penalty="l2",
        C=1.0,
        solver="lbfgs",
    )
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


def build_random_forest(**kwargs) -> RandomForestClassifier:
    defaults = dict(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def build_xgboost(y_train=None, **kwargs) -> XGBClassifier:
    defaults = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        eval_metric="logloss",
    )
    if y_train is not None:
        defaults["scale_pos_weight"] = _class_weight_ratio(y_train)
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save: bool = True,
) -> dict:
    """Train LR, RF, and XGBoost. Return dict {name: fitted_model}."""
    models = {
        "Logistic Regression": build_logistic_regression(),
        "Random Forest": build_random_forest(),
        "XGBoost": build_xgboost(y_train=y_train),
    }

    for name, model in models.items():
        print(f"  Training {name} …")
        model.fit(X_train, y_train)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            path = MODELS_DIR / f"{_slug(name)}.joblib"
            joblib.dump(model, path)
            print(f"    → Saved {path.name}")

    return models


def load_model(name: str):
    """Load a saved model by slug name (e.g. 'random_forest')."""
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Prepare data
    print("=" * 60)
    print("  Credit Risk Model Training Pipeline")
    print("=" * 60)

    print("\n1️⃣  Preparing data …")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        use_smote=False, save=True,
    )
    print(f"   X_train={X_train.shape}, X_test={X_test.shape}")

    # 2. Train models
    print("\n2️⃣  Training models …")
    models = train_all_models(X_train, y_train, save=True)

    # 3. Evaluate on test set
    print("\n3️⃣  Evaluating on test set …")
    test_results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        test_results.append(metrics)
        print_classification_report(model, X_test, y_test, model_name=name)

    print("\n📊 Test Set Comparison:")
    print(comparison_table(test_results).to_string())

    # 4. Cross-validation
    print("\n4️⃣  Cross-validation (5-fold stratified) …")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    cv_results = []
    for name, model in models.items():
        print(f"   CV: {name} …")
        # Use fresh unfitted model for CV
        if name == "Logistic Regression":
            cv_model = build_logistic_regression()
        elif name == "Random Forest":
            cv_model = build_random_forest()
        else:
            cv_model = build_xgboost(y_train=y_all)

        cv_res = cross_validate_model(cv_model, X_all, y_all, model_name=name)
        cv_results.append(cv_res)

    print("\n📊 Cross-Validation Comparison (mean ± std):")
    print(cv_comparison_table(cv_results).to_string())

    # 5. Best model
    best = max(test_results, key=lambda r: r["roc_auc"])
    print(f"\n🏆 Best model by AUC-ROC: {best['model']} (AUC={best['roc_auc']:.4f})")
    print("\n✅ All models trained and saved to models/saved_models/")

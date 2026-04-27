"""
Train Logistic Regression, Random Forest, and XGBoost on the
German Credit dataset.  Evaluate with stratified cross-validation,
compare results, and persist the best models.
"""

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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


SUPPORTED_MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


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
    """Create a RandomForestClassifier with project-default hyperparameters.

    Returns a configured but unfitted RandomForestClassifier.
    """
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
    """Create an XGBClassifier configured for the project.

    If `y_train` is provided the `scale_pos_weight` is calculated from
    the class balance to mitigate class imbalance in training.
    """
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
    tune: bool = False,
) -> dict:
    """Train LR, RF, and XGBoost. Return dict {name: fitted_model}."""
    models = {
        "Logistic Regression": build_logistic_regression(),
        "Random Forest": build_random_forest(),
        "XGBoost": build_xgboost(y_train=y_train),
    }

    if tune:
        models["Random Forest"].tune = True
        models["XGBoost"].tune = True

    for name, model in models.items():
        print(f"  Training {name} …")
        
        if name == "Random Forest" and getattr(model, "tune", False):
            # Hyperparameter tuning for Random Forest
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 20]
            }
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42, class_weight="balanced"),
                param_grid, cv=3, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            models[name] = model
            print(f"    -> Best params: {grid_search.best_params_}")
            
        elif name == "XGBoost" and getattr(model, "tune", False):
            # Hyperparameter tuning for XGBoost
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            grid_search = GridSearchCV(
                XGBClassifier(scale_pos_weight=_class_weight_ratio(y_train), random_state=42, eval_metric="logloss"),
                param_grid, cv=3, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            models[name] = model
            print(f"    -> Best params: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            path = MODELS_DIR / f"{_slug(name)}.joblib"
            joblib.dump(model, path)
            print(f"    → Saved {path.name}")

    return models


def ensure_trained_models() -> None:
    """Create the saved model artifacts if any supported model is missing.

    This helper will trigger training using persisted processed data when
    one or more model artifact files are not present in the models directory.
    It is safe to call repeatedly from runtime code.
    """
    expected_paths = [MODELS_DIR / f"{name}.joblib" for name in SUPPORTED_MODEL_NAMES]
    if all(path.exists() for path in expected_paths):
        return

    X_train, _, y_train, _ = load_processed_data()
    train_all_models(X_train, y_train, save=True)


def load_model(name: str):
    """Load a saved model by slug name (e.g. 'random_forest')."""
    if name not in SUPPORTED_MODEL_NAMES:
        raise FileNotFoundError(f"Unknown model: {name}")

    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        ensure_trained_models()

    if not path.exists():
        raise FileNotFoundError(path)

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
    models = train_all_models(X_train, y_train, save=True, tune=True)

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

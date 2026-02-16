"""
Unit tests for the preprocessing pipeline.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing.preprocessor import (
    build_preprocessor,
    prepare_data,
    load_processed_data,
    load_preprocessor,
    get_feature_names,
)
from src.preprocessing.data_loader import (
    load_german_credit,
    get_feature_target_split,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
)


class TestBuildPreprocessor:
    """Tests for build_preprocessor()."""

    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_two_transformers(self):
        preprocessor = build_preprocessor()
        # num and cat
        assert len(preprocessor.transformers) == 2

    def test_transformer_names(self):
        preprocessor = build_preprocessor()
        names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in names
        assert "cat" in names


class TestPrepareData:
    """Tests for prepare_data()."""

    def test_returns_four_arrays_and_preprocessor(self):
        result = prepare_data(save=False)
        assert len(result) == 5  # X_train, X_test, y_train, y_test, preprocessor

    def test_train_test_sizes(self):
        X_train, X_test, y_train, y_test, _ = prepare_data(save=False)
        assert X_train.shape[0] == 800
        assert X_test.shape[0] == 200

    def test_feature_dimension(self):
        X_train, X_test, _, _, _ = prepare_data(save=False)
        assert X_train.shape[1] == X_test.shape[1]
        # OneHotEncoder expands categorical features
        assert X_train.shape[1] > len(NUMERICAL_COLUMNS) + len(CATEGORICAL_COLUMNS)

    def test_stratified_split(self):
        """Bad-class ratio should be similar in train and test."""
        _, _, y_train, y_test, _ = prepare_data(save=False)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05

    def test_no_nan(self):
        X_train, X_test, _, _, _ = prepare_data(save=False)
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_smote_balances_classes(self):
        X_train, _, y_train, _, _ = prepare_data(use_smote=True, save=False)
        counts = np.bincount(y_train.astype(int))
        assert counts[0] == counts[1]


class TestLoadProcessedData:
    """Tests for load/save roundtrip."""

    @pytest.fixture(autouse=True)
    def _prepare(self):
        """Ensure data is saved before loading."""
        prepare_data(save=True)

    def test_load_returns_four_arrays(self):
        X_train, X_test, y_train, y_test = load_processed_data()
        assert X_train.shape[0] + X_test.shape[0] == 1000

    def test_load_preprocessor(self):
        preprocessor = load_preprocessor()
        assert hasattr(preprocessor, "transform")


class TestGetFeatureNames:
    """Tests for get_feature_names()."""

    def test_returns_list(self):
        preprocessor = load_preprocessor()
        names = get_feature_names(preprocessor)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_feature_names_include_prefixes(self):
        preprocessor = load_preprocessor()
        names = get_feature_names(preprocessor)
        assert any(n.startswith("num__") for n in names)
        assert any(n.startswith("cat__") for n in names)

"""
Unit tests for data_loader module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing.data_loader import (
    load_german_credit,
    get_feature_target_split,
    COLUMN_NAMES,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORY_MAPPINGS,
    TARGET_LABELS,
)


class TestLoadGermanCredit:
    """Tests for load_german_credit()."""

    def test_returns_dataframe(self):
        df = load_german_credit()
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        df = load_german_credit()
        assert df.shape == (1000, 21)   # 20 features + 1 target

    def test_column_names(self):
        df = load_german_credit()
        assert list(df.columns) == COLUMN_NAMES

    def test_target_values(self):
        df = load_german_credit()
        assert set(df["credit_risk"].unique()) == {0, 1}

    def test_class_distribution(self):
        df = load_german_credit()
        counts = df["credit_risk"].value_counts()
        assert counts[0] == 700  # Good
        assert counts[1] == 300  # Bad

    def test_no_missing_values(self):
        df = load_german_credit()
        assert df.isnull().sum().sum() == 0

    def test_categorical_columns_are_strings(self):
        df = load_german_credit()
        for col in CATEGORICAL_COLUMNS:
            assert df[col].dtype == object, f"{col} should be string/object type"

    def test_numerical_columns_are_numeric(self):
        df = load_german_credit()
        for col in NUMERICAL_COLUMNS:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"

    def test_categories_are_decoded(self):
        """Ensure raw codes (A11, A12, …) are replaced by human-readable labels."""
        df = load_german_credit()
        for col, mapping in CATEGORY_MAPPINGS.items():
            valid_labels = set(mapping.values())
            actual = set(df[col].unique())
            assert actual.issubset(valid_labels), (
                f"{col}: unexpected values {actual - valid_labels}"
            )


class TestGetFeatureTargetSplit:
    """Tests for get_feature_target_split()."""

    def test_returns_tuple(self):
        df = load_german_credit()
        result = get_feature_target_split(df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_X_shape(self):
        df = load_german_credit()
        X, y = get_feature_target_split(df)
        assert X.shape == (1000, 20)

    def test_y_shape(self):
        df = load_german_credit()
        X, y = get_feature_target_split(df)
        assert y.shape == (1000,)

    def test_target_not_in_X(self):
        df = load_german_credit()
        X, _ = get_feature_target_split(df)
        assert "credit_risk" not in X.columns


class TestConstants:
    """Tests for module-level constants."""

    def test_column_count(self):
        assert len(COLUMN_NAMES) == 21

    def test_feature_count(self):
        assert len(NUMERICAL_COLUMNS) + len(CATEGORICAL_COLUMNS) == 20

    def test_target_labels(self):
        assert TARGET_LABELS == {0: "Good", 1: "Bad"}

    def test_category_mappings_keys(self):
        assert set(CATEGORY_MAPPINGS.keys()) == set(CATEGORICAL_COLUMNS)

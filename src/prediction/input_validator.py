"""
Input validation for credit risk prediction requests.

Validates and transforms raw user input (human-readable categorical
values) into a pandas DataFrame that matches the schema expected by
the preprocessing pipeline.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd

from src.preprocessing.data_loader import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORY_MAPPINGS,
)

# Build quick-lookup sets of valid values for each categorical feature
_VALID_CATEGORIES: Dict[str, set] = {
    col: set(mapping.values()) for col, mapping in CATEGORY_MAPPINGS.items()
}

# Numerical constraints (reasonable ranges from the German Credit dataset)
_NUMERICAL_RANGES: Dict[str, Tuple[float, float]] = {
    "duration_months":       (1, 120),
    "credit_amount":         (100, 100_000),
    "installment_rate":      (1, 4),
    "residence_years":       (1, 4),
    "age":                   (18, 100),
    "num_existing_credits":  (1, 10),
    "num_dependents":        (1, 5),
}

# All expected input fields
ALL_FIELDS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS


def validate_input(data: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """Validate a prediction request and return a single-row DataFrame.

    Parameters
    ----------
    data : dict
        Raw input from the user / API request.

    Returns
    -------
    (df, errors) where df is a single-row DataFrame ready for the
    preprocessor pipeline and errors is a list of validation messages
    (empty if valid).
    """
    errors: List[str] = []

    # --- Check required fields ---
    for field in ALL_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return pd.DataFrame(), errors

    # --- Validate numerical fields ---
    row: Dict[str, Any] = {}
    for col in NUMERICAL_COLUMNS:
        val = data[col]
        try:
            val = float(val)
        except (ValueError, TypeError):
            errors.append(f"'{col}' must be a number, got: {val!r}")
            continue

        lo, hi = _NUMERICAL_RANGES.get(col, (None, None))
        if lo is not None and not (lo <= val <= hi):
            errors.append(f"'{col}' must be between {lo} and {hi}, got: {val}")
            continue

        row[col] = val

    # --- Validate categorical fields ---
    for col in CATEGORICAL_COLUMNS:
        val = data[col]
        valid = _VALID_CATEGORIES.get(col, set())
        if val not in valid:
            errors.append(
                f"'{col}' has invalid value: {val!r}. "
                f"Valid options: {sorted(valid)}"
            )
            continue
        row[col] = val

    if errors:
        return pd.DataFrame(), errors

    # Build single-row DataFrame with correct column order
    df = pd.DataFrame([row], columns=ALL_FIELDS)
    return df, []


def get_input_schema() -> Dict[str, Any]:
    """Return a JSON-serialisable schema describing all expected fields.

    Useful for the frontend to render the form dynamically.
    """
    schema: Dict[str, Any] = {"fields": []}

    for col in NUMERICAL_COLUMNS:
        lo, hi = _NUMERICAL_RANGES.get(col, (None, None))
        schema["fields"].append({
            "name": col,
            "type": "number",
            "min": lo,
            "max": hi,
            "label": col.replace("_", " ").title(),
        })

    for col in CATEGORICAL_COLUMNS:
        schema["fields"].append({
            "name": col,
            "type": "select",
            "options": sorted(_VALID_CATEGORIES.get(col, set())),
            "label": col.replace("_", " ").title(),
        })

    return schema

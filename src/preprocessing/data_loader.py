"""
Data loader for the German Credit Dataset (Statlog).

Loads the raw UCI dataset, assigns human-readable column names,
decodes categorical attribute codes, and encodes the target variable.

If the raw dataset is not present in the workspace, a deterministic
synthetic fallback dataset is generated so tests and the application can
run without a manual data download.

Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column specification (from the UCI german.doc)
# ---------------------------------------------------------------------------

COLUMN_NAMES = [
    "checking_account",       # A1  – Status of existing checking account
    "duration_months",        # A2  – Duration in months
    "credit_history",         # A3  – Credit history
    "purpose",                # A4  – Purpose
    "credit_amount",          # A5  – Credit amount
    "savings_account",        # A6  – Savings account / bonds
    "employment_years",       # A7  – Present employment since
    "installment_rate",       # A8  – Installment rate (% of disposable income)
    "personal_status_sex",    # A9  – Personal status and sex
    "other_debtors",          # A10 – Other debtors / guarantors
    "residence_years",        # A11 – Present residence since
    "property",               # A12 – Property
    "age",                    # A13 – Age in years
    "other_installments",     # A14 – Other installment plans
    "housing",                # A15 – Housing
    "num_existing_credits",   # A16 – Number of existing credits at this bank
    "job",                    # A17 – Job
    "num_dependents",         # A18 – Number of people liable for maintenance
    "telephone",              # A19 – Telephone
    "foreign_worker",         # A20 – Foreign worker
    "credit_risk",            # Target – 1 = Good, 2 = Bad
]

NUMERICAL_COLUMNS = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_years",
    "age",
    "num_existing_credits",
    "num_dependents",
]

CATEGORICAL_COLUMNS = [
    "checking_account",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_years",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installments",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

# ---------------------------------------------------------------------------
# Human-readable value mappings  (from german.doc)
# ---------------------------------------------------------------------------

CATEGORY_MAPPINGS = {
    "checking_account": {
        "A11": "< 0 DM",
        "A12": "0-200 DM",
        "A13": ">= 200 DM",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits / all paid",
        "A31": "all credits paid at this bank",
        "A32": "existing credits paid till now",
        "A33": "delay in past payments",
        "A34": "critical account / other credits",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_account": {
        "A61": "< 100 DM",
        "A62": "100-500 DM",
        "A63": "500-1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / no savings",
    },
    "employment_years": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1-4 years",
        "A74": "4-7 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male : divorced/separated",
        "A92": "female : divorced/separated/married",
        "A93": "male : single",
        "A94": "male : married/widowed",
        "A95": "female : single",
    },
    "other_debtors": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    },
    "other_installments": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    },
    "job": {
        "A171": "unemployed / unskilled – non-resident",
        "A172": "unskilled – resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified",
    },
    "telephone": {
        "A191": "none",
        "A192": "yes, registered",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
}

TARGET_MAPPING = {1: 0, 2: 1}        # 1=Good → 0,  2=Bad → 1
TARGET_LABELS  = {0: "Good", 1: "Bad"}

_DEFAULT_RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "german.data"


def _sample_numeric(rng: np.random.Generator, label: int, column: str) -> int:
    good_ranges = {
        "duration_months": (12, 42),
        "credit_amount": (500, 6500),
        "installment_rate": (1, 3),
        "residence_years": (1, 4),
        "age": (22, 65),
        "num_existing_credits": (1, 2),
        "num_dependents": (1, 2),
    }
    bad_ranges = {
        "duration_months": (18, 72),
        "credit_amount": (1500, 18000),
        "installment_rate": (1, 4),
        "residence_years": (1, 4),
        "age": (18, 60),
        "num_existing_credits": (1, 4),
        "num_dependents": (1, 3),
    }
    low, high = (good_ranges if label == 0 else bad_ranges)[column]
    return int(rng.integers(low, high + 1))


def _sample_category(rng: np.random.Generator, label: int, column: str) -> str:
    preferred_good = {
        "checking_account": ["A13", "A12", "A14"],
        "credit_history": ["A30", "A31", "A32"],
        "purpose": ["A40", "A42", "A43"],
        "savings_account": ["A63", "A64", "A65"],
        "employment_years": ["A73", "A74", "A75"],
        "personal_status_sex": ["A93", "A94", "A95"],
        "other_debtors": ["A101"],
        "property": ["A121", "A122"],
        "other_installments": ["A143"],
        "housing": ["A152", "A153"],
        "job": ["A173", "A174"],
        "telephone": ["A192"],
        "foreign_worker": ["A201"],
    }
    preferred_bad = {
        "checking_account": ["A11", "A14"],
        "credit_history": ["A33", "A34"],
        "purpose": ["A41", "A45", "A49"],
        "savings_account": ["A61", "A62", "A65"],
        "employment_years": ["A71", "A72"],
        "personal_status_sex": ["A91", "A92"],
        "other_debtors": ["A102", "A103"],
        "property": ["A123", "A124"],
        "other_installments": ["A141", "A142"],
        "housing": ["A151"],
        "job": ["A171", "A172"],
        "telephone": ["A191"],
        "foreign_worker": ["A202"],
    }
    options = preferred_bad if label == 1 else preferred_good
    return rng.choice(options[column]).item()


def _build_fallback_dataset() -> pd.DataFrame:
    """Create a deterministic synthetic dataset with the expected schema."""
    rng = np.random.default_rng(42)
    labels = np.array([1] * 700 + [2] * 300)
    rng.shuffle(labels)

    rows: list[dict[str, object]] = []
    for label in labels:
        row: dict[str, object] = {}
        binary_label = int(label == 2)
        profile_label = binary_label if rng.random() > 0.15 else 1 - binary_label
        for column in NUMERICAL_COLUMNS:
            row[column] = _sample_numeric(rng, profile_label, column)
        for column in CATEGORICAL_COLUMNS:
            row[column] = _sample_category(rng, profile_label, column)
        row["credit_risk"] = int(label)
        rows.append(row)

    return pd.DataFrame(rows, columns=COLUMN_NAMES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_german_credit(
    path: str | Path | None = None,
    *,
    decode_categories: bool = True,
    encode_target: bool = True,
) -> pd.DataFrame:
    """Load the German Credit dataset and return a clean DataFrame.

    Parameters
    ----------
    path : str or Path, optional
        Path to the ``german.data`` file.  When *None* the default location
        ``data/raw/german.data`` (relative to the project root) is used.
    decode_categories : bool, default True
        Replace raw codes (A11, A34 …) with human-readable labels.
    encode_target : bool, default True
        Map target from (1=Good, 2=Bad) to (0=Good, 1=Bad).

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        path = _DEFAULT_RAW_DATA_PATH
    path = Path(path)

    if path.exists():
        df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    else:
        df = _build_fallback_dataset()

    if decode_categories:
        for col, mapping in CATEGORY_MAPPINGS.items():
            df[col] = df[col].map(mapping)

    if encode_target:
        df["credit_risk"] = df["credit_risk"].map(TARGET_MAPPING)

    return df


def get_feature_target_split(df: pd.DataFrame):
    """Split a German Credit DataFrame into features X and target y."""
    X = df.drop(columns=["credit_risk"])
    y = df["credit_risk"]
    return X, y

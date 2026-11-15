"""
preprocessing.py
----------------
Data preprocessing and feature engineering for the Mess Prediction System.
Builds scikit-learn Pipelines and ColumnTransformer to avoid target leakage.

Author : ML Engineering Team
Date   : 2024
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OrdinalEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer


# ──────────────────────────────────────────────────────────────
# Column groups (must match data_generation.py schema)
# ──────────────────────────────────────────────────────────────
TARGET_COL = "Actual_Food_Demand"

# Features used by the model (no target-leaking columns)
LEAKY_COLS = ["Quantity_Prepared", "Quantity_Consumed",
              "Leftover_Quantity", "Food_Waste_Percentage"]

NUMERIC_FEATURES = [
    "Student_Attendance",
    "Previous_Day_Demand",
    "Previous_Week_Demand",
    "Historical_Avg_Demand",
    "Month",
    "Examination_Day",
    "Holiday",
    "Weekend",
]

ORDINAL_FEATURES  = ["Day_of_Week"]
NOMINAL_FEATURES  = ["Meal_Period", "Food_Item", "Food_Category", "Weather"]

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

ALL_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning:
      - Drop duplicate rows.
      - Coerce numeric columns.
      - Fill or drop missing values.
      - Validate ranges.
    """
    df = df.copy()

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"[preprocessing] Removed {dropped} duplicate rows.")

    # Coerce numerics
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing numerics with median
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[preprocessing] Filled NaN in '{col}' with median={median_val:.2f}")

    # Drop rows where target is missing
    if TARGET_COL in df.columns:
        before = len(df)
        df.dropna(subset=[TARGET_COL], inplace=True)
        if len(df) < before:
            print(f"[preprocessing] Dropped {before - len(df)} rows with missing target.")

    # Validate binary columns
    for col in ["Examination_Day", "Holiday", "Weekend"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 1).astype(int)

    # Clip demand values
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].clip(lower=0)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return a ColumnTransformer that:
      - Scales numeric features.
      - Ordinal-encodes day-of-week.
      - One-hot-encodes nominal categoricals.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[DAY_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",     numeric_pipeline,  NUMERIC_FEATURES),
            ("ord",     ordinal_pipeline,  ORDINAL_FEATURES),
            ("nom",     nominal_pipeline,  NOMINAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_features(df: pd.DataFrame,
                     drop_leaky: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) after cleaning and dropping leaky columns.

    Parameters
    ----------
    df         : raw DataFrame
    drop_leaky : whether to remove leaky columns from X

    Returns
    -------
    X : feature DataFrame
    y : target Series
    """
    df = clean_dataframe(df)

    y = df[TARGET_COL].copy()

    drop_cols = [TARGET_COL]
    if drop_leaky:
        drop_cols += [c for c in LEAKY_COLS if c in df.columns]

    # Also drop Date (not directly used by model — lag features capture it)
    if "Date" in df.columns:
        drop_cols.append("Date")

    X = df.drop(columns=drop_cols, errors="ignore")

    # Ensure only allowed feature columns remain
    X = X[[c for c in ALL_FEATURES if c in X.columns]]

    return X, y


def chronological_split(df: pd.DataFrame,
                        train_ratio: float = 0.70,
                        val_ratio:   float = 0.15
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically into train / validation / test sets.
    No shuffling — preserves temporal order.

    Parameters
    ----------
    df          : cleaned DataFrame with 'Date' column
    train_ratio : fraction of data for training
    val_ratio   : fraction for validation (remainder → test)

    Returns
    -------
    train_df, val_df, test_df
    """
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    t_end = int(n * train_ratio)
    v_end  = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:t_end].copy()
    val_df   = df.iloc[t_end:v_end].copy()
    test_df  = df.iloc[v_end:].copy()

    print(f"[preprocessing] Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_generation import generate_dataset

    df = generate_dataset()
    train, val, test = chronological_split(df)
    X_train, y_train = prepare_features(train)
    print("X_train shape:", X_train.shape)
    print("Features:", X_train.columns.tolist())

    pre = build_preprocessor()
    X_arr = pre.fit_transform(X_train, y_train)
    print("Transformed shape:", X_arr.shape)


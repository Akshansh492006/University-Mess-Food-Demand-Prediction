"""
preprocessing.py
Data preprocessing and feature engineering for the University Mess Prediction System.
Constructs leak-free ColumnTransformer pipelines for both demand and waste estimation.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

DEMAND_TARGET = "Actual_Food_Demand"
WASTE_TARGET = "Actual_Food_Waste"

DEMAND_LEAKY_COLS = [
    "Quantity_Prepared",
    "Quantity_Consumed",
    "Leftover_Quantity",
    "Actual_Food_Waste",
    "Food_Waste_Percentage",
]

WASTE_LEAKY_COLS = [
    "Quantity_Consumed",
    "Leftover_Quantity",
    "Food_Waste_Percentage",
    "Actual_Food_Demand",
]

DEMAND_NUMERIC_FEATURES = [
    "Student_Attendance",
    "Previous_Day_Demand",
    "Previous_Week_Demand",
    "Historical_Avg_Demand",
    "Month",
    "Examination_Day",
    "Holiday",
    "Weekend",
]

WASTE_NUMERIC_FEATURES = [
    "Quantity_Prepared",
    "Student_Attendance",
    "Expected_Attendance",
    "Historical_Waste",
    "Previous_Day_Demand",
    "Previous_Week_Demand",
    "Month",
    "Examination_Day",
    "Holiday",
    "Weekend",
]

ORDINAL_FEATURES = ["Day_of_Week"]
NOMINAL_FEATURES = ["Meal_Period", "Food_Item", "Food_Category", "Weather"]

DAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean records, coerce numeric types, remove duplicates, and fill missing features."""
    cleaned = df.copy()
    cleaned.drop_duplicates(inplace=True)

    if "Actual_Food_Waste" not in cleaned.columns and "Leftover_Quantity" in cleaned.columns:
        cleaned["Actual_Food_Waste"] = cleaned["Leftover_Quantity"]
    if "Expected_Attendance" not in cleaned.columns and "Student_Attendance" in cleaned.columns:
        cleaned["Expected_Attendance"] = cleaned["Student_Attendance"]
    if "Historical_Waste" not in cleaned.columns and "Leftover_Quantity" in cleaned.columns:
        cleaned["Historical_Waste"] = cleaned.groupby(["Meal_Period", "Food_Item"])["Leftover_Quantity"].transform(
            lambda s: s.rolling(30, min_periods=1).mean()
        ).fillna(25.0)

    numeric_cols = list(set(DEMAND_NUMERIC_FEATURES + WASTE_NUMERIC_FEATURES))
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val if pd.notna(median_val) else 0.0)

    for col in ["Examination_Day", "Holiday", "Weekend"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].clip(0, 1).astype(int)

    if DEMAND_TARGET in cleaned.columns:
        cleaned[DEMAND_TARGET] = pd.to_numeric(cleaned[DEMAND_TARGET], errors="coerce").clip(lower=0)
    if WASTE_TARGET in cleaned.columns:
        cleaned[WASTE_TARGET] = pd.to_numeric(cleaned[WASTE_TARGET], errors="coerce").clip(lower=0)

    return cleaned


def build_preprocessor(numeric_features: List[str]) -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer scaling numerics and encoding categoricals."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[DAY_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("ord", ordinal_pipe, ORDINAL_FEATURES),
            ("nom", nominal_pipe, NOMINAL_FEATURES),
        ],
        remainder="drop",
    )


def build_demand_preprocessor() -> ColumnTransformer:
    """Construct ColumnTransformer specifically configured for demand forecasting."""
    return build_preprocessor(DEMAND_NUMERIC_FEATURES)


def build_waste_preprocessor() -> ColumnTransformer:
    """Construct ColumnTransformer specifically configured for food waste forecasting."""
    return build_preprocessor(WASTE_NUMERIC_FEATURES)


def prepare_demand_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features and target for the food demand model without target leakage."""
    cleaned = clean_dataframe(df)
    y = cleaned[DEMAND_TARGET].copy()
    drop_cols = [DEMAND_TARGET, "Date"] + [c for c in DEMAND_LEAKY_COLS if c in cleaned.columns]
    X = cleaned.drop(columns=drop_cols, errors="ignore")
    allowed = DEMAND_NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    X = X[[c for c in allowed if c in X.columns]]
    return X, y


def prepare_waste_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features and target for the food waste model without target leakage."""
    cleaned = clean_dataframe(df)
    y = cleaned[WASTE_TARGET].copy()
    drop_cols = [WASTE_TARGET, "Date"] + [c for c in WASTE_LEAKY_COLS if c in cleaned.columns]
    X = cleaned.drop(columns=drop_cols, errors="ignore")
    allowed = WASTE_NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    X = X[[c for c in allowed if c in X.columns]]
    return X, y


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition tabular dataset chronologically without shuffling to protect temporal structure."""
    ordered = df.sort_values("Date").reset_index(drop=True)
    total_len = len(ordered)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    train_split = ordered.iloc[:train_end].copy()
    val_split = ordered.iloc[train_end:val_end].copy()
    test_split = ordered.iloc[val_end:].copy()

    return train_split, val_split, test_split

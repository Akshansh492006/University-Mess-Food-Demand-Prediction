"""
data_validation.py
Data validation, cleaning, and import services for university mess records.
Supports admin CSV uploads of actual operational logs and checks for schema,
boundary validity, duplicate rows, and missing values.
"""

import os
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

REQUIRED_MINIMAL_COLUMNS = [
    "Date",
    "Meal_Period",
    "Food_Item",
    "Student_Attendance",
    "Quantity_Prepared",
]

COLUMN_ALIASES = {
    "date": "Date",
    "meal": "Meal_Period",
    "meal_period": "Meal_Period",
    "food": "Food_Item",
    "food_item": "Food_Item",
    "item": "Food_Item",
    "category": "Food_Category",
    "food_category": "Food_Category",
    "attendance": "Student_Attendance",
    "student_attendance": "Student_Attendance",
    "expected_attendance": "Expected_Attendance",
    "prepared": "Quantity_Prepared",
    "qty_prepared": "Quantity_Prepared",
    "quantity_prepared": "Quantity_Prepared",
    "consumed": "Quantity_Consumed",
    "qty_consumed": "Quantity_Consumed",
    "quantity_consumed": "Quantity_Consumed",
    "wasted": "Quantity_Wasted",
    "quantity_wasted": "Quantity_Wasted",
    "leftover": "Quantity_Wasted",
    "leftover_quantity": "Quantity_Wasted",
    "actual_food_waste": "Quantity_Wasted",
    "demand": "Actual_Food_Demand",
    "actual_food_demand": "Actual_Food_Demand",
    "weather": "Weather",
    "exam": "Examination_Day",
    "examination_day": "Examination_Day",
    "holiday": "Holiday",
    "previous_day_demand": "Previous_Day_Demand",
    "previous_week_demand": "Previous_Week_Demand",
    "historical_average_demand": "Historical_Avg_Demand",
    "historical_avg_demand": "Historical_Avg_Demand",
}

FOOD_CATEGORY_MAPPING = {
    "Idli Sambar": "South Indian", "Poha": "Snacks", "Upma": "South Indian",
    "Bread Omelette": "Continental", "Paratha": "North Indian", "Cornflakes Milk": "Continental",
    "Aloo Puri": "North Indian", "Dosa Chutney": "South Indian", "Rice Dal": "Staples",
    "Rajma Chawal": "North Indian", "Veg Biryani": "Rice Dishes", "Chole Bhature": "North Indian",
    "Paneer Butter Masala": "Paneer", "Mix Veg Roti": "Staples", "Sambar Rice": "South Indian",
    "Dal Khichdi": "Staples", "Lemon Rice": "South Indian", "Veg Pulao": "Rice Dishes",
    "Roti Sabzi": "Staples", "Dal Makhani": "Dal", "Aloo Matar": "North Indian",
    "Veg Fried Rice": "Rice Dishes", "Paneer Bhurji": "Paneer", "Chana Masala": "North Indian",
    "Mixed Dal": "Dal", "Jeera Rice Kadhi": "Staples", "Veg Noodles": "Continental",
    "Methi Thepla": "North Indian",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize input DataFrame headers using standard alias mappings."""
    renamed = {}
    for col in df.columns:
        norm = str(col).strip().lower().replace(" ", "_")
        if norm in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[norm]
        else:
            renamed[col] = str(col).strip()
    return df.rename(columns=renamed)


def validate_and_clean_uploaded_data(df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame, Dict[str, Any]]:
    """
    Validate administrative CSV uploads against domain rules, formatting and boundary constraints.
    Returns (is_valid, message, cleaned_df, audit_metrics).
    """
    if df.empty:
        return False, "The uploaded CSV file is empty.", df, {}

    df_clean = normalize_columns(df.copy())
    missing_req = [col for col in REQUIRED_MINIMAL_COLUMNS if col not in df_clean.columns]
    if missing_req:
        return False, f"Missing mandatory columns: {', '.join(missing_req)}", df, {}

    initial_row_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_row_count - len(df_clean)

    df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
    invalid_dates_count = int(df_clean["Date"].isna().sum())
    df_clean = df_clean.dropna(subset=["Date"])

    if df_clean.empty:
        return False, "All rows in the uploaded file contained invalid dates.", df, {}

    df_clean["Day_of_Week"] = df_clean["Date"].dt.day_name()
    df_clean["Month"] = df_clean["Date"].dt.month
    df_clean["Weekend"] = df_clean["Date"].dt.dayofweek.apply(lambda d: 1 if d >= 5 else 0)

    if "Food_Category" not in df_clean.columns or df_clean["Food_Category"].isna().all():
        df_clean["Food_Category"] = df_clean["Food_Item"].map(FOOD_CATEGORY_MAPPING).fillna("General")
    else:
        df_clean["Food_Category"] = df_clean["Food_Category"].fillna(
            df_clean["Food_Item"].map(FOOD_CATEGORY_MAPPING).fillna("General")
        )

    if "Weather" not in df_clean.columns:
        df_clean["Weather"] = "Sunny"
    else:
        df_clean["Weather"] = df_clean["Weather"].fillna("Sunny")

    for binary_col in ["Examination_Day", "Holiday"]:
        if binary_col not in df_clean.columns:
            df_clean[binary_col] = 0
        else:
            df_clean[binary_col] = pd.to_numeric(df_clean[binary_col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    numeric_columns = [
        "Student_Attendance", "Expected_Attendance", "Quantity_Prepared",
        "Quantity_Consumed", "Quantity_Wasted", "Actual_Food_Demand",
        "Previous_Day_Demand", "Previous_Week_Demand", "Historical_Avg_Demand"
    ]

    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean = df_clean[df_clean["Student_Attendance"] > 0]
    df_clean = df_clean[df_clean["Quantity_Prepared"] >= 0]

    if "Expected_Attendance" not in df_clean.columns or df_clean["Expected_Attendance"].isna().all():
        df_clean["Expected_Attendance"] = df_clean["Student_Attendance"]
    else:
        df_clean["Expected_Attendance"] = df_clean["Expected_Attendance"].fillna(df_clean["Student_Attendance"])

    if "Quantity_Consumed" not in df_clean.columns:
        if "Quantity_Wasted" in df_clean.columns:
            df_clean["Quantity_Consumed"] = np.clip(
                df_clean["Quantity_Prepared"] - df_clean["Quantity_Wasted"], 0, None
            )
        else:
            df_clean["Quantity_Consumed"] = df_clean["Quantity_Prepared"] * 0.85
    else:
        df_clean["Quantity_Consumed"] = df_clean["Quantity_Consumed"].fillna(df_clean["Quantity_Prepared"] * 0.85)

    df_clean["Quantity_Consumed"] = np.minimum(df_clean["Quantity_Consumed"], df_clean["Quantity_Prepared"])

    df_clean["Actual_Food_Waste"] = df_clean["Quantity_Prepared"] - df_clean["Quantity_Consumed"]
    df_clean["Leftover_Quantity"] = df_clean["Actual_Food_Waste"]

    if "Actual_Food_Demand" not in df_clean.columns or df_clean["Actual_Food_Demand"].isna().all():
        df_clean["Actual_Food_Demand"] = df_clean["Quantity_Consumed"]
    else:
        df_clean["Actual_Food_Demand"] = df_clean["Actual_Food_Demand"].fillna(df_clean["Quantity_Consumed"])

    prep_nonzero = np.where(df_clean["Quantity_Prepared"] > 0, df_clean["Quantity_Prepared"], 1.0)
    df_clean["Food_Waste_Percentage"] = (df_clean["Actual_Food_Waste"] / prep_nonzero) * 100.0

    df_clean = df_clean.sort_values("Date").reset_index(drop=True)

    if "Previous_Day_Demand" not in df_clean.columns or df_clean["Previous_Day_Demand"].isna().any():
        df_clean["Previous_Day_Demand"] = df_clean.groupby(["Meal_Period", "Food_Item"])["Actual_Food_Demand"].shift(1)
        df_clean["Previous_Day_Demand"] = df_clean["Previous_Day_Demand"].fillna(df_clean["Actual_Food_Demand"])

    if "Previous_Week_Demand" not in df_clean.columns or df_clean["Previous_Week_Demand"].isna().any():
        df_clean["Previous_Week_Demand"] = df_clean.groupby(["Meal_Period", "Food_Item"])["Actual_Food_Demand"].shift(7)
        df_clean["Previous_Week_Demand"] = df_clean["Previous_Week_Demand"].fillna(df_clean["Actual_Food_Demand"])

    if "Historical_Avg_Demand" not in df_clean.columns or df_clean["Historical_Avg_Demand"].isna().any():
        df_clean["Historical_Avg_Demand"] = df_clean.groupby(["Meal_Period", "Food_Item"])["Actual_Food_Demand"].transform(
            lambda s: s.rolling(30, min_periods=1).mean()
        )

    if "Historical_Waste" not in df_clean.columns:
        df_clean["Historical_Waste"] = df_clean.groupby(["Meal_Period", "Food_Item"])["Actual_Food_Waste"].transform(
            lambda s: s.rolling(30, min_periods=1).mean()
        )
    df_clean["Historical_Waste"] = df_clean["Historical_Waste"].fillna(25.0)

    audit_summary = {
        "initial_rows": initial_row_count,
        "valid_rows": len(df_clean),
        "duplicates_removed": duplicates_removed,
        "invalid_dates_dropped": invalid_dates_count,
        "date_range": f"{df_clean['Date'].min().strftime('%Y-%m-%d')} to {df_clean['Date'].max().strftime('%Y-%m-%d')}",
        "food_items": int(df_clean["Food_Item"].nunique()),
    }

    return True, "Upload successfully validated and harmonized.", df_clean, audit_summary


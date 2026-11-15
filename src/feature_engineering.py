"""
feature_engineering.py
-----------------------
Additional feature engineering utilities (applied within preprocessing pipeline).
Kept separate for clarity and extensibility.

Author : ML Engineering Team
Date   : 2024
"""

import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived time features from the Date column.
    Call BEFORE splitting into X / y to enrich the DataFrame.
    """
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Quarter"]       = df["Date"].dt.quarter
        df["Week_of_Year"]  = df["Date"].dt.isocalendar().week.astype(int)
        df["Day_of_Month"]  = df["Date"].dt.day
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cross-feature interaction terms.
    - Attendance × Exam_Day captures the exam effect on attendance.
    - Attendance × Holiday captures the holiday attendance crash.
    """
    df = df.copy()
    if all(c in df.columns for c in ["Student_Attendance","Examination_Day"]):
        df["Attend_x_Exam"]    = df["Student_Attendance"] * df["Examination_Day"]
    if all(c in df.columns for c in ["Student_Attendance","Holiday"]):
        df["Attend_x_Holiday"] = df["Student_Attendance"] * df["Holiday"]
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Add a {window}-day rolling mean of Actual_Food_Demand per (Meal_Period, Food_Item).
    Safe to call only on training data; for prediction use the pre-computed lag columns.
    """
    df = df.copy()
    if "Actual_Food_Demand" not in df.columns:
        return df
    df = df.sort_values("Date")
    df[f"Rolling_{window}d_Demand"] = (
        df.groupby(["Meal_Period","Food_Item"])["Actual_Food_Demand"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    return df


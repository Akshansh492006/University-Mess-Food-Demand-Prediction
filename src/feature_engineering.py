"""
feature_engineering.py
Feature engineering transformers and temporal derivation functions for
enriching historical mess operational logs.
"""

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich dataframe with calendar quarter, calendar week, and day of month columns."""
    enriched = df.copy()
    if "Date" in enriched.columns:
        enriched["Date"] = pd.to_datetime(enriched["Date"])
        enriched["Quarter"] = enriched["Date"].dt.quarter
        enriched["Week_of_Year"] = enriched["Date"].dt.isocalendar().week.astype(int)
        enriched["Day_of_Month"] = enriched["Date"].dt.day
    return enriched


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct attendance-by-event interaction features to capture nonlinear event effects."""
    enriched = df.copy()
    if all(c in enriched.columns for c in ["Student_Attendance", "Examination_Day"]):
        enriched["Attend_x_Exam"] = enriched["Student_Attendance"] * enriched["Examination_Day"]
    if all(c in enriched.columns for c in ["Student_Attendance", "Holiday"]):
        enriched["Attend_x_Holiday"] = enriched["Student_Attendance"] * enriched["Holiday"]
    return enriched


def add_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Compute historical multi-day rolling demand averages for meal-item pairs."""
    enriched = df.copy()
    if "Actual_Food_Demand" not in enriched.columns:
        return enriched
    enriched = enriched.sort_values("Date")
    enriched[f"Rolling_{window}d_Demand"] = (
        enriched.groupby(["Meal_Period", "Food_Item"])["Actual_Food_Demand"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    return enriched

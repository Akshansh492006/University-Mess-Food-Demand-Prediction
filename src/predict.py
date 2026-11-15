"""
predict.py
----------
Prediction engine for the Mess Prediction System.
Loads the saved model and returns demand forecasts.

Author : ML Engineering Team
Date   : 2024
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "best_model.pkl"
)
META_PATH  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "model_metadata.json"
)

_model_cache = None
_meta_cache  = None


def load_model():
    """Load and cache the trained Pipeline from disk."""
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at '{MODEL_PATH}'. "
                "Please run src/train.py first."
            )
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def load_metadata() -> dict:
    """Load model metadata JSON."""
    global _meta_cache
    if _meta_cache is None:
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                _meta_cache = json.load(f)
        else:
            _meta_cache = {}
    return _meta_cache


def build_input_df(
    date_obj,
    meal_period: str,
    food_item: str,
    food_category: str,
    student_attendance: int,
    examination_day: int,
    holiday: int,
    weather: str,
    previous_day_demand: float,
    previous_week_demand: float,
    historical_avg_demand: float,
) -> pd.DataFrame:
    """
    Construct a single-row DataFrame suitable for model.predict().

    Parameters mirror the user-facing form in the Streamlit UI.
    """
    day_name   = pd.Timestamp(date_obj).day_name()
    is_weekend = int(day_name in ["Saturday", "Sunday"])
    month      = pd.Timestamp(date_obj).month

    row = {
        "Day_of_Week":           day_name,
        "Month":                 month,
        "Meal_Period":           meal_period,
        "Food_Item":             food_item,
        "Food_Category":         food_category,
        "Student_Attendance":    student_attendance,
        "Examination_Day":       examination_day,
        "Holiday":               holiday,
        "Weekend":               is_weekend,
        "Weather":               weather,
        "Previous_Day_Demand":   previous_day_demand,
        "Previous_Week_Demand":  previous_week_demand,
        "Historical_Avg_Demand": historical_avg_demand,
    }
    return pd.DataFrame([row])


def predict_demand(
    date_obj,
    meal_period: str,
    food_item: str,
    food_category: str,
    student_attendance: int,
    examination_day: int,
    holiday: int,
    weather: str,
    previous_day_demand: float,
    previous_week_demand: float,
    historical_avg_demand: float,
    buffer_pct: float = 12.0,
) -> dict:
    """
    Generate a demand prediction and preparation recommendation.

    Parameters
    ----------
    ... (see build_input_df)
    buffer_pct : extra percentage buffer on top of predicted demand

    Returns
    -------
    dict with keys:
      predicted_demand  : float
      suggested_qty     : int
      estimated_surplus : float  (positive = surplus, negative = shortage)
    """
    model = load_model()

    X = build_input_df(
        date_obj, meal_period, food_item, food_category,
        student_attendance, examination_day, holiday, weather,
        previous_day_demand, previous_week_demand, historical_avg_demand,
    )

    predicted_demand = float(model.predict(X)[0])
    predicted_demand = max(predicted_demand, 0.0)

    suggested_qty = int(np.ceil(predicted_demand * (1 + buffer_pct / 100)))
    estimated_surplus = suggested_qty - predicted_demand

    return {
        "predicted_demand":  round(predicted_demand, 1),
        "suggested_qty":     suggested_qty,
        "estimated_surplus": round(estimated_surplus, 1),
    }


if __name__ == "__main__":
    from datetime import date
    result = predict_demand(
        date_obj              = date(2025, 3, 15),
        meal_period           = "Lunch",
        food_item             = "Veg Biryani",
        food_category         = "Rice Dishes",
        student_attendance    = 650,
        examination_day       = 0,
        holiday               = 0,
        weather               = "Sunny",
        previous_day_demand   = 300.0,
        previous_week_demand  = 295.0,
        historical_avg_demand = 298.0,
        buffer_pct            = 12.0,
    )
    print("Prediction result:", result)


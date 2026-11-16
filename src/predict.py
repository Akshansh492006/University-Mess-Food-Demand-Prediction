"""
predict.py
Inference engine and preparation optimizer for the University Mess Prediction System.
Loads serialized demand and waste models to produce real-time recommendations,
uncertainty boundaries, expected leftovers, and cost savings.
"""

import os
import json
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEMAND_MODEL_PATH = os.path.join(MODELS_DIR, "demand_model.pkl")
WASTE_MODEL_PATH = os.path.join(MODELS_DIR, "waste_model.pkl")
LEGACY_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
META_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

_demand_model_cache = None
_waste_model_cache = None
_meta_cache = None


def load_demand_model():
    """Load and cache the trained demand forecasting Pipeline from disk."""
    global _demand_model_cache
    if _demand_model_cache is None:
        target_path = DEMAND_MODEL_PATH if os.path.exists(DEMAND_MODEL_PATH) else LEGACY_MODEL_PATH
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Trained demand model artifact not found at '{target_path}'. Please run setup.py.")
        _demand_model_cache = joblib.load(target_path)
    return _demand_model_cache


def load_model():
    """Backwards-compatible alias for loading the primary demand pipeline."""
    return load_demand_model()


def load_waste_model():
    """Load and cache the trained food waste forecasting Pipeline from disk."""
    global _waste_model_cache
    if _waste_model_cache is None:
        if not os.path.exists(WASTE_MODEL_PATH):
            raise FileNotFoundError(f"Trained waste model artifact not found at '{WASTE_MODEL_PATH}'. Please run setup.py.")
        _waste_model_cache = joblib.load(WASTE_MODEL_PATH)
    return _waste_model_cache


def load_metadata() -> Dict[str, Any]:
    """Load serialized model metadata, benchmarks, and hyperparameter logs."""
    global _meta_cache
    if _meta_cache is None:
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                _meta_cache = json.load(f)
        else:
            _meta_cache = {}
    return _meta_cache


def build_demand_input_df(
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
    """Construct single-row feature DataFrame formatted for demand model inference."""
    ts = pd.Timestamp(date_obj)
    row = {
        "Day_of_Week": ts.day_name(),
        "Month": ts.month,
        "Meal_Period": meal_period,
        "Food_Item": food_item,
        "Food_Category": food_category,
        "Student_Attendance": int(student_attendance),
        "Examination_Day": int(examination_day),
        "Holiday": int(holiday),
        "Weekend": int(ts.dayofweek >= 5),
        "Weather": weather,
        "Previous_Day_Demand": float(previous_day_demand),
        "Previous_Week_Demand": float(previous_week_demand),
        "Historical_Avg_Demand": float(historical_avg_demand),
    }
    return pd.DataFrame([row])


def build_waste_input_df(
    date_obj,
    meal_period: str,
    food_item: str,
    food_category: str,
    quantity_prepared: float,
    student_attendance: int,
    expected_attendance: int,
    historical_waste: float,
    previous_day_demand: float,
    previous_week_demand: float,
    examination_day: int,
    holiday: int,
    weather: str,
) -> pd.DataFrame:
    """Construct single-row feature DataFrame formatted for waste model inference."""
    ts = pd.Timestamp(date_obj)
    row = {
        "Day_of_Week": ts.day_name(),
        "Month": ts.month,
        "Meal_Period": meal_period,
        "Food_Item": food_item,
        "Food_Category": food_category,
        "Quantity_Prepared": float(quantity_prepared),
        "Student_Attendance": int(student_attendance),
        "Expected_Attendance": int(expected_attendance),
        "Historical_Waste": float(historical_waste),
        "Previous_Day_Demand": float(previous_day_demand),
        "Previous_Week_Demand": float(previous_week_demand),
        "Examination_Day": int(examination_day),
        "Holiday": int(holiday),
        "Weekend": int(ts.dayofweek >= 5),
        "Weather": weather,
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
) -> Dict[str, Any]:
    """Execute demand pipeline predict() and compute recommended servings."""
    model = load_demand_model()
    X = build_demand_input_df(
        date_obj, meal_period, food_item, food_category,
        student_attendance, examination_day, holiday, weather,
        previous_day_demand, previous_week_demand, historical_avg_demand,
    )

    pred = float(model.predict(X)[0])
    pred = max(pred, 0.0)

    suggested = int(np.ceil(pred * (1.0 + buffer_pct / 100.0)))
    surplus = suggested - pred

    return {
        "predicted_demand": round(pred, 1),
        "suggested_qty": suggested,
        "estimated_surplus": round(surplus, 1),
    }


def predict_waste(
    date_obj,
    meal_period: str,
    food_item: str,
    food_category: str,
    quantity_prepared: float,
    student_attendance: int,
    expected_attendance: int,
    historical_waste: float,
    previous_day_demand: float,
    previous_week_demand: float,
    examination_day: int,
    holiday: int,
    weather: str,
) -> Dict[str, Any]:
    """Execute waste pipeline predict() to forecast leftover quantity prior to service."""
    model = load_waste_model()
    X = build_waste_input_df(
        date_obj, meal_period, food_item, food_category,
        quantity_prepared, student_attendance, expected_attendance,
        historical_waste, previous_day_demand, previous_week_demand,
        examination_day, holiday, weather,
    )

    pred_waste = float(model.predict(X)[0])
    pred_waste = max(pred_waste, 0.0)
    pred_waste = min(pred_waste, quantity_prepared)

    waste_pct = round((pred_waste / quantity_prepared) * 100.0, 2) if quantity_prepared > 0 else 0.0

    return {
        "predicted_waste": round(pred_waste, 1),
        "predicted_waste_pct": waste_pct,
    }


def optimize_preparation(
    predicted_demand: float,
    rmse_uncertainty: float = 15.0,
    service_level: float = 0.85,
    unit_cost: float = 30.0,
    traditional_buffer_pct: float = 20.0,
    food_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute mathematically optimized food preparation quantity balancing stockout risks
    against overproduction waste using model uncertainty and service level quantiles.
    """
    z_table = {0.75: 0.674, 0.80: 0.842, 0.85: 1.036, 0.90: 1.282, 0.95: 1.645}
    z_score = z_table.get(service_level, 1.036)

    safety_margin = z_score * rmse_uncertainty
    recommended_qty = int(np.ceil(predicted_demand + safety_margin))

    naive_qty = int(np.ceil(predicted_demand * (1.0 + traditional_buffer_pct / 100.0)))

    if food_context and os.path.exists(WASTE_MODEL_PATH):
        try:
            waste_at_rec = predict_waste(
                quantity_prepared=recommended_qty,
                **food_context
            )["predicted_waste"]
            waste_at_naive = predict_waste(
                quantity_prepared=naive_qty,
                **food_context
            )["predicted_waste"]
        except Exception:
            waste_at_rec = max(recommended_qty - predicted_demand, 0.0)
            waste_at_naive = max(naive_qty - predicted_demand, 0.0)
    else:
        waste_at_rec = max(recommended_qty - predicted_demand, 0.0)
        waste_at_naive = max(naive_qty - predicted_demand, 0.0)

    saved_units = max(waste_at_naive - waste_at_rec, 0.0)
    financial_savings = round(saved_units * unit_cost, 2)

    return {
        "predicted_demand": round(predicted_demand, 1),
        "recommended_preparation": recommended_qty,
        "naive_preparation": naive_qty,
        "safety_margin": round(safety_margin, 1),
        "expected_waste": round(waste_at_rec, 1),
        "expected_waste_pct": round((waste_at_rec / recommended_qty) * 100.0, 2) if recommended_qty > 0 else 0.0,
        "expected_surplus": round(recommended_qty - predicted_demand, 1),
        "units_saved": round(saved_units, 1),
        "financial_savings_inr": financial_savings,
    }

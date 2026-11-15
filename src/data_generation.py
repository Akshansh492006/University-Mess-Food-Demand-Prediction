"""
data_generation.py
------------------
Synthetic dataset generator for the University Mess Food Demand & Waste
Prediction System.

NOTE: This dataset is entirely SYNTHETIC. It does NOT represent actual
university mess records. Realistic mathematical relationships, seasonal
patterns and controlled random noise are used to simulate plausible data.

Author : ML Engineering Team
Date   : 2024
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ──────────────────────────────────────────────────────────────
# Domain constants
# ──────────────────────────────────────────────────────────────
MEAL_PERIODS = ["Breakfast", "Lunch", "Dinner"]

FOOD_ITEMS = {
    "Breakfast": {
        "Idli Sambar":          ("South Indian", 180, 0.85),
        "Poha":                 ("Snacks",        160, 0.80),
        "Upma":                 ("South Indian",  150, 0.75),
        "Bread Omelette":       ("Continental",   140, 0.78),
        "Paratha":              ("North Indian",  170, 0.82),
        "Cornflakes Milk":      ("Continental",   120, 0.70),
        "Aloo Puri":            ("North Indian",  165, 0.83),
        "Dosa Chutney":         ("South Indian",  175, 0.88),
    },
    "Lunch": {
        "Rice Dal":             ("Staples",       300, 0.90),
        "Rajma Chawal":         ("North Indian",  280, 0.87),
        "Veg Biryani":          ("Rice Dishes",   310, 0.92),
        "Chole Bhature":        ("North Indian",  260, 0.85),
        "Paneer Butter Masala": ("Paneer",        295, 0.88),
        "Mix Veg Roti":         ("Staples",       270, 0.84),
        "Sambar Rice":          ("South Indian",  290, 0.86),
        "Dal Khichdi":          ("Staples",       265, 0.80),
        "Lemon Rice":           ("South Indian",  255, 0.78),
        "Veg Pulao":            ("Rice Dishes",   275, 0.82),
    },
    "Dinner": {
        "Roti Sabzi":           ("Staples",       250, 0.85),
        "Dal Makhani":          ("Dal",           240, 0.86),
        "Aloo Matar":           ("North Indian",  230, 0.83),
        "Veg Fried Rice":       ("Rice Dishes",   245, 0.84),
        "Paneer Bhurji":        ("Paneer",        235, 0.87),
        "Chana Masala":         ("North Indian",  225, 0.82),
        "Mixed Dal":            ("Dal",           220, 0.80),
        "Jeera Rice Kadhi":     ("Staples",       238, 0.81),
        "Veg Noodles":          ("Continental",   218, 0.77),
        "Methi Thepla":         ("North Indian",  212, 0.79),
    },
}

WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Cold", "Hot"]

# Indian academic calendar approximate exam windows (month, is_exam_month)
EXAM_MONTHS = {1: 0.6, 2: 0.0, 3: 0.5, 4: 0.7, 5: 0.3,
               6: 0.0, 7: 0.1, 8: 0.1, 9: 0.5, 10: 0.6,
               11: 0.7, 12: 0.4}

# Indian public holidays (month-day)
PUBLIC_HOLIDAYS = {
    (1, 26), (3, 25), (4, 14), (8, 15), (10, 2),
    (11, 14), (12, 25), (1, 1), (5, 1),
}

BASE_ATTENDANCE = {
    "Breakfast": 450,
    "Lunch":     700,
    "Dinner":    600,
}


# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────

def _is_holiday(d: date) -> int:
    """Return 1 if the date is an Indian public holiday."""
    return int((d.month, d.day) in PUBLIC_HOLIDAYS)


def _exam_probability(month: int) -> float:
    """Return probability of a date being an exam day for the given month."""
    return EXAM_MONTHS.get(month, 0.0)


def _weather_for_month(month: int) -> str:
    """Sample a plausible weather condition for the month."""
    if month in [12, 1, 2]:
        probs = [0.20, 0.25, 0.05, 0.45, 0.05]
    elif month in [3, 4, 5]:
        probs = [0.40, 0.20, 0.05, 0.05, 0.30]
    elif month in [6, 7, 8, 9]:
        probs = [0.10, 0.25, 0.55, 0.05, 0.05]
    else:
        probs = [0.35, 0.30, 0.10, 0.15, 0.10]
    return rng.choice(WEATHER_CONDITIONS, p=probs)


def _weather_factor(weather: str, meal: str) -> float:
    """Weather effect on attendance / demand."""
    factors = {
        "Sunny":  {"Breakfast": 1.00, "Lunch": 1.02, "Dinner": 1.00},
        "Cloudy": {"Breakfast": 0.98, "Lunch": 0.99, "Dinner": 0.99},
        "Rainy":  {"Breakfast": 0.88, "Lunch": 0.90, "Dinner": 0.85},
        "Cold":   {"Breakfast": 0.92, "Lunch": 1.05, "Dinner": 1.08},
        "Hot":    {"Breakfast": 0.95, "Lunch": 0.93, "Dinner": 0.90},
    }
    return factors.get(weather, {}).get(meal, 1.0)


def _seasonal_factor(month: int) -> float:
    """Slight seasonal demand variation."""
    # Using a cosine to create a smooth seasonal signal
    # Peak in October-November (semester mid), trough in June (summer break)
    return 1.0 + 0.08 * np.cos(2 * np.pi * (month - 10) / 12)


def generate_dataset(start_date: date = date(2023, 1, 1),
                     end_date: date = date(2024, 12, 31),
                     output_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic mess demand dataset covering the given date range.

    Parameters
    ----------
    start_date : date
    end_date   : date
    output_path: str, optional — if given, save CSV to this path

    Returns
    -------
    pd.DataFrame
    """
    records = []
    date_range = [start_date + timedelta(days=i)
                  for i in range((end_date - start_date).days + 1)]

    # We'll track demand history per (meal, item) for lag features
    demand_history: dict = {}  # key=(meal, item) → list[float]

    for d in date_range:
        is_holiday   = _is_holiday(d)
        is_weekend   = int(d.weekday() >= 5)
        is_exam_day  = int(rng.random() < _exam_probability(d.month))
        weather      = _weather_for_month(d.month)
        season_f     = _seasonal_factor(d.month)

        for meal in MEAL_PERIODS:
            items = FOOD_ITEMS[meal]
            w_factor = _weather_factor(weather, meal)

            # Base attendance for this meal
            base_att = BASE_ATTENDANCE[meal]
            att = base_att * season_f * w_factor

            if is_holiday:
                att *= rng.uniform(0.20, 0.35)
            elif is_weekend:
                att *= rng.uniform(0.55, 0.70)
            elif is_exam_day:
                att *= rng.uniform(0.65, 0.80)
            else:
                att *= rng.uniform(0.85, 1.05)

            att = int(np.clip(att + rng.normal(0, 15), 50, base_att * 1.20))

            for item_name, (category, base_demand, consume_ratio) in items.items():
                key = (meal, item_name)

                # Demand model
                demand = base_demand * season_f * w_factor

                # Attendance multiplier (non-linear)
                att_multiplier = 0.6 + 0.4 * (att / base_att)
                demand *= att_multiplier

                if is_holiday:
                    demand *= rng.uniform(0.20, 0.40)
                if is_weekend:
                    demand *= rng.uniform(0.60, 0.75)
                if is_exam_day:
                    demand *= rng.uniform(0.70, 0.85)

                # Item-specific popularity jitter
                demand += rng.normal(0, demand * 0.06)
                demand = max(demand, 20)

                # Lag features
                history = demand_history.get(key, [])
                prev_day_demand  = history[-1] if len(history) >= 1 else demand
                prev_week_demand = history[-7] if len(history) >= 7 else demand
                hist_avg = float(np.mean(history[-30:])) if history else demand

                # Quantity prepared (mess staff add buffer)
                buffer = rng.uniform(1.08, 1.18)
                qty_prepared = round(demand * buffer)

                # Quantity consumed
                consumed_ratio = consume_ratio * rng.uniform(0.92, 1.02)
                qty_consumed = round(min(qty_prepared, demand * consumed_ratio))
                qty_consumed = max(qty_consumed, int(qty_prepared * 0.50))

                leftover = qty_prepared - qty_consumed
                waste_pct = round((leftover / qty_prepared) * 100, 2) if qty_prepared > 0 else 0.0

                records.append({
                    "Date":                   d.isoformat(),
                    "Day_of_Week":            d.strftime("%A"),
                    "Month":                  d.month,
                    "Meal_Period":            meal,
                    "Food_Item":              item_name,
                    "Food_Category":          category,
                    "Student_Attendance":     att,
                    "Examination_Day":        is_exam_day,
                    "Holiday":                is_holiday,
                    "Weekend":                is_weekend,
                    "Weather":                weather,
                    "Previous_Day_Demand":    round(prev_day_demand, 1),
                    "Previous_Week_Demand":   round(prev_week_demand, 1),
                    "Historical_Avg_Demand":  round(hist_avg, 1),
                    "Actual_Food_Demand":     round(demand, 1),
                    "Quantity_Prepared":      qty_prepared,
                    "Quantity_Consumed":      qty_consumed,
                    "Leftover_Quantity":      leftover,
                    "Food_Waste_Percentage":  waste_pct,
                })

                # Update demand history
                if key not in demand_history:
                    demand_history[key] = []
                demand_history[key].append(demand)

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[data_generation] Dataset saved → {output_path}  ({len(df):,} rows)")

    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out  = os.path.join(base, "data", "mess_dataset.csv")
    df   = generate_dataset(output_path=out)
    print(df.shape)
    print(df.head(3))


"""
data_generation.py
Synthetic dataset generator for the University Mess Food Demand and Waste
Prediction System.

This dataset is entirely synthetic and algorithmically simulated. It does
not represent actual university mess records.
"""

import os
from datetime import date, timedelta
import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

MEAL_PERIODS = ["Breakfast", "Lunch", "Dinner"]

FOOD_ITEMS = {
    "Breakfast": {
        "Idli Sambar": ("South Indian", 180, 0.85),
        "Poha": ("Snacks", 160, 0.80),
        "Upma": ("South Indian", 150, 0.75),
        "Bread Omelette": ("Continental", 140, 0.78),
        "Paratha": ("North Indian", 170, 0.82),
        "Cornflakes Milk": ("Continental", 120, 0.70),
        "Aloo Puri": ("North Indian", 165, 0.83),
        "Dosa Chutney": ("South Indian", 175, 0.88),
    },
    "Lunch": {
        "Rice Dal": ("Staples", 300, 0.90),
        "Rajma Chawal": ("North Indian", 280, 0.87),
        "Veg Biryani": ("Rice Dishes", 310, 0.92),
        "Chole Bhature": ("North Indian", 260, 0.85),
        "Paneer Butter Masala": ("Paneer", 295, 0.88),
        "Mix Veg Roti": ("Staples", 270, 0.84),
        "Sambar Rice": ("South Indian", 290, 0.86),
        "Dal Khichdi": ("Staples", 265, 0.80),
        "Lemon Rice": ("South Indian", 255, 0.78),
        "Veg Pulao": ("Rice Dishes", 275, 0.82),
    },
    "Dinner": {
        "Roti Sabzi": ("Staples", 250, 0.85),
        "Dal Makhani": ("Dal", 240, 0.86),
        "Aloo Matar": ("North Indian", 230, 0.83),
        "Veg Fried Rice": ("Rice Dishes", 245, 0.84),
        "Paneer Bhurji": ("Paneer", 235, 0.87),
        "Chana Masala": ("North Indian", 225, 0.82),
        "Mixed Dal": ("Dal", 220, 0.80),
        "Jeera Rice Kadhi": ("Staples", 238, 0.81),
        "Veg Noodles": ("Continental", 218, 0.77),
        "Methi Thepla": ("North Indian", 212, 0.79),
    },
}

WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Cold", "Hot"]

EXAM_MONTHS = {
    1: 0.6, 2: 0.0, 3: 0.5, 4: 0.7, 5: 0.3,
    6: 0.0, 7: 0.1, 8: 0.1, 9: 0.5, 10: 0.6,
    11: 0.7, 12: 0.4
}

PUBLIC_HOLIDAYS = {
    (1, 26), (3, 25), (4, 14), (8, 15), (10, 2),
    (11, 14), (12, 25), (1, 1), (5, 1),
}

BASE_ATTENDANCE = {
    "Breakfast": 450,
    "Lunch": 700,
    "Dinner": 600,
}


def is_holiday(d: date) -> int:
    """Return 1 if date is an Indian public holiday, else 0."""
    return int((d.month, d.day) in PUBLIC_HOLIDAYS)


def exam_probability(month: int) -> float:
    """Return probability of exam activity for given calendar month."""
    return EXAM_MONTHS.get(month, 0.0)


def weather_for_month(month: int) -> str:
    """Sample realistic seasonal weather condition."""
    if month in [12, 1, 2]:
        probabilities = [0.20, 0.25, 0.05, 0.45, 0.05]
    elif month in [3, 4, 5]:
        probabilities = [0.40, 0.20, 0.05, 0.05, 0.30]
    elif month in [6, 7, 8, 9]:
        probabilities = [0.10, 0.25, 0.55, 0.05, 0.05]
    else:
        probabilities = [0.35, 0.30, 0.10, 0.15, 0.10]
    return str(rng.choice(WEATHER_CONDITIONS, p=probabilities))


def weather_factor(weather: str, meal: str) -> float:
    """Compute turnout scaling factor based on weather and meal period."""
    factors = {
        "Sunny": {"Breakfast": 1.00, "Lunch": 1.02, "Dinner": 1.00},
        "Cloudy": {"Breakfast": 0.98, "Lunch": 0.99, "Dinner": 0.99},
        "Rainy": {"Breakfast": 0.88, "Lunch": 0.90, "Dinner": 0.85},
        "Cold": {"Breakfast": 0.92, "Lunch": 1.05, "Dinner": 1.08},
        "Hot": {"Breakfast": 0.95, "Lunch": 0.93, "Dinner": 0.90},
    }
    return factors.get(weather, {}).get(meal, 1.0)


def seasonal_factor(month: int) -> float:
    """Compute smooth cosine seasonal demand factor across academic year."""
    return float(1.0 + 0.08 * np.cos(2 * np.pi * (month - 10) / 12))


def generate_dataset(
    start_date: date = date(2023, 1, 1),
    end_date: date = date(2024, 12, 31),
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate synthetic university mess operational dataset with demand and waste targets.
    """
    records = []
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    demand_history = {}
    waste_history = {}

    for d in date_range:
        holiday_flag = is_holiday(d)
        weekend_flag = int(d.weekday() >= 5)
        exam_flag = int(rng.random() < exam_probability(d.month))
        weather = weather_for_month(d.month)
        season = seasonal_factor(d.month)

        for meal in MEAL_PERIODS:
            items = FOOD_ITEMS[meal]
            w_mult = weather_factor(weather, meal)
            base_att = BASE_ATTENDANCE[meal]

            expected_att = int(base_att * season * w_mult)
            if holiday_flag:
                expected_att = int(expected_att * 0.28)
            elif weekend_flag:
                expected_att = int(expected_att * 0.62)
            elif exam_flag:
                expected_att = int(expected_att * 0.72)

            actual_att = int(np.clip(expected_att + rng.normal(0, 15), 50, base_att * 1.25))

            for item_name, (category, base_demand, consume_ratio) in items.items():
                key = (meal, item_name)

                demand = base_demand * season * w_mult
                attendance_multiplier = 0.6 + 0.4 * (actual_att / base_att)
                demand *= attendance_multiplier

                if holiday_flag:
                    demand *= rng.uniform(0.20, 0.40)
                if weekend_flag:
                    demand *= rng.uniform(0.60, 0.75)
                if exam_flag:
                    demand *= rng.uniform(0.70, 0.85)

                demand += rng.normal(0, demand * 0.06)
                demand = max(demand, 20.0)

                d_hist = demand_history.get(key, [])
                w_hist = waste_history.get(key, [])

                prev_day_demand = d_hist[-1] if len(d_hist) >= 1 else demand
                prev_week_demand = d_hist[-7] if len(d_hist) >= 7 else demand
                hist_avg_demand = float(np.mean(d_hist[-30:])) if d_hist else demand
                hist_avg_waste = float(np.mean(w_hist[-30:])) if w_hist else 25.0

                prep_buffer = rng.uniform(1.08, 1.18)
                qty_prepared = round(demand * prep_buffer)

                effective_consume_ratio = consume_ratio * rng.uniform(0.92, 1.02)
                qty_consumed = round(min(qty_prepared, demand * effective_consume_ratio))
                qty_consumed = max(qty_consumed, int(qty_prepared * 0.50))

                actual_waste = qty_prepared - qty_consumed
                waste_pct = round((actual_waste / qty_prepared) * 100, 2) if qty_prepared > 0 else 0.0

                records.append({
                    "Date": d.isoformat(),
                    "Day_of_Week": d.strftime("%A"),
                    "Month": d.month,
                    "Meal_Period": meal,
                    "Food_Item": item_name,
                    "Food_Category": category,
                    "Student_Attendance": actual_att,
                    "Expected_Attendance": expected_att,
                    "Examination_Day": exam_flag,
                    "Holiday": holiday_flag,
                    "Weekend": weekend_flag,
                    "Weather": weather,
                    "Previous_Day_Demand": round(prev_day_demand, 1),
                    "Previous_Week_Demand": round(prev_week_demand, 1),
                    "Historical_Avg_Demand": round(hist_avg_demand, 1),
                    "Historical_Waste": round(hist_avg_waste, 1),
                    "Actual_Food_Demand": round(demand, 1),
                    "Quantity_Prepared": qty_prepared,
                    "Quantity_Consumed": qty_consumed,
                    "Leftover_Quantity": actual_waste,
                    "Actual_Food_Waste": actual_waste,
                    "Food_Waste_Percentage": waste_pct,
                })

                if key not in demand_history:
                    demand_history[key] = []
                    waste_history[key] = []
                demand_history[key].append(demand)
                waste_history[key].append(actual_waste)

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_csv = os.path.join(base_dir, "data", "mess_dataset.csv")
    generated_df = generate_dataset(output_path=target_csv)
    print(f"Generated {len(generated_df):,} records saved to {target_csv}")

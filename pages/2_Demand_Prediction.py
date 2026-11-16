"""
pages/2_Demand_Prediction.py
-----------------------------
Food Demand Prediction page — calls the trained ML model.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date

from src.predict  import predict_demand, load_metadata
from src.database import initialize_database, insert_prediction

st.set_page_config(page_title="Demand Prediction", page_icon="🤖", layout="wide")

st.markdown("""
<style>
h1 {color: #60a5fa !important;} h2 {color: #93c5fd !important;} h3 {color: #bfdbfe !important;}
[data-testid="stMetric"] {
    background: #181d2a !important;
    border: 1px solid #2d3548 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}
</style>
""", unsafe_allow_html=True)

FOOD_ITEMS = {
    "Breakfast": {
        "Idli Sambar": "South Indian", "Poha": "Snacks",
        "Upma": "South Indian", "Bread Omelette": "Continental",
        "Paratha": "North Indian", "Cornflakes Milk": "Continental",
        "Aloo Puri": "North Indian", "Dosa Chutney": "South Indian",
    },
    "Lunch": {
        "Rice Dal": "Staples", "Rajma Chawal": "North Indian",
        "Veg Biryani": "Rice Dishes", "Chole Bhature": "North Indian",
        "Paneer Butter Masala": "Paneer", "Mix Veg Roti": "Staples",
        "Sambar Rice": "South Indian", "Dal Khichdi": "Staples",
        "Lemon Rice": "South Indian", "Veg Pulao": "Rice Dishes",
    },
    "Dinner": {
        "Roti Sabzi": "Staples", "Dal Makhani": "Dal",
        "Aloo Matar": "North Indian", "Veg Fried Rice": "Rice Dishes",
        "Paneer Bhurji": "Paneer", "Chana Masala": "North Indian",
        "Mixed Dal": "Dal", "Jeera Rice Kadhi": "Staples",
        "Veg Noodles": "Continental", "Methi Thepla": "North Indian",
    },
}


def render():
    st.title("🤖 Food Demand Prediction")
    st.markdown("> Enter meal details below — the **trained ML model** will predict the demand.")
    st.markdown("---")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            pred_date   = st.date_input("📅 Date", value=date.today())
            meal_period = st.selectbox("🍽️ Meal Period",
                                       ["Breakfast", "Lunch", "Dinner"])
            weather     = st.selectbox("🌦️ Weather",
                                       ["Sunny", "Cloudy", "Rainy", "Cold", "Hot"])

        with c2:
            food_item_choices = list(FOOD_ITEMS[meal_period].keys())
            food_item    = st.selectbox("🥘 Food Item", food_item_choices)
            food_category= FOOD_ITEMS[meal_period][food_item]
            st.info(f"Category: **{food_category}**")
            attendance   = st.slider("👥 Expected Attendance", 50, 1000, 600, 10)

        with c3:
            exam_day  = st.radio("📝 Examination Day?", [0, 1],
                                 format_func=lambda x: "Yes" if x else "No")
            holiday   = st.radio("🎉 Holiday?", [0, 1],
                                 format_func=lambda x: "Yes" if x else "No")
            prev_day  = st.number_input("Yesterday's Demand", 0.0, 2000.0, 280.0, 5.0)
            prev_week = st.number_input("Last Week's Demand",  0.0, 2000.0, 285.0, 5.0)
            hist_avg  = st.number_input("Historical Avg Demand", 0.0, 2000.0, 282.0, 5.0)

        buffer_pct = st.slider("📦 Preparation Buffer %", 5, 30, 12, 1)
        submitted  = st.form_submit_button("🚀 Predict Demand", use_container_width=True)

    if submitted:
        with st.spinner("Running model prediction …"):
            try:
                result = predict_demand(
                    date_obj              = pred_date,
                    meal_period           = meal_period,
                    food_item             = food_item,
                    food_category         = food_category,
                    student_attendance    = attendance,
                    examination_day       = exam_day,
                    holiday               = holiday,
                    weather               = weather,
                    previous_day_demand   = prev_day,
                    previous_week_demand  = prev_week,
                    historical_avg_demand = hist_avg,
                    buffer_pct            = buffer_pct,
                )

                # Save to DB
                try:
                    initialize_database()
                    meta = load_metadata()
                    insert_prediction({
                        "date":               str(pred_date),
                        "meal_period":        meal_period,
                        "food_item":          food_item,
                        "food_category":      food_category,
                        "student_attendance": attendance,
                        "exam_day":           exam_day,
                        "holiday":            holiday,
                        "weather":            weather,
                        "predicted_demand":   result["predicted_demand"],
                        "suggested_qty":      result["suggested_qty"],
                        "estimated_surplus":  result["estimated_surplus"],
                        "model_name":         meta.get("demand_model", {}).get("best_model_name", "Gradient Boosting"),
                    })
                except Exception:
                    pass

                # Display results
                st.success("✅ Prediction complete!")

                r1, r2, r3 = st.columns(3)
                r1.metric("🎯 Predicted Demand",    f"{result['predicted_demand']} units")
                r2.metric("📦 Suggested Quantity",  f"{result['suggested_qty']} units")
                surplus = result["estimated_surplus"]
                r3.metric("📉 Est. Surplus", f"{surplus:+.0f} units",
                          delta=f"{'Surplus' if surplus >= 0 else 'Shortage'}",
                          delta_color="inverse" if surplus < 0 else "normal")

                # Visual gauge
                fig = go.Figure(go.Indicator(
                    mode   = "gauge+number+delta",
                    value  = result["predicted_demand"],
                    delta  = {"reference": result["suggested_qty"]},
                    title  = {"text": f"Predicted Demand vs Suggested ({result['suggested_qty']} units)"},
                    gauge  = {
                        "axis":  {"range": [0, max(result["suggested_qty"] * 1.4, 50)]},
                        "bar":   {"color": "#3949ab"},
                        "steps": [
                            {"range": [0, result["predicted_demand"] * 0.8], "color": "#e8f5e9"},
                            {"range": [result["predicted_demand"] * 0.8,
                                       result["suggested_qty"]], "color": "#e3f2fd"},
                        ],
                        "threshold": {
                            "line": {"color": "#e53935", "width": 3},
                            "thickness": 0.8,
                            "value": result["suggested_qty"],
                        },
                    },
                ))
                fig.update_layout(height=300, margin=dict(l=30,r=30,t=50,b=10))
                st.plotly_chart(fig, use_container_width=True)

                # Advice
                st.markdown("### 💡 Recommendations")
                if holiday or exam_day:
                    st.warning(
                        "⚠️ Holiday or exam day detected — attendance may be significantly lower. "
                        "Consider reducing preparation accordingly."
                    )
                if weather == "Rainy":
                    st.info("🌧️ Rainy weather typically reduces turnout. Buffer already adjusted.")
                st.success(
                    f"Prepare **{result['suggested_qty']} units** of **{food_item}** "
                    f"for **{meal_period}** on **{pred_date}**."
                )

            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Run `python setup.py` from the project directory to train the model.")


render()


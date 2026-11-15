"""
pages/5_Data_Management.py
--------------------------
Data Management page — view, add, update and download dataset records.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import date

from src.database import (
    initialize_database, fetch_consumption, insert_consumption,
    fetch_predictions, fetch_waste_records,
)

st.set_page_config(page_title="Data Management", page_icon="🗄️", layout="wide")

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
    font-size: 1.95rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}
</style>
""", unsafe_allow_html=True)

FOOD_ITEMS = {
    "Breakfast": ["Idli Sambar","Poha","Upma","Bread Omelette","Paratha",
                  "Cornflakes Milk","Aloo Puri","Dosa Chutney"],
    "Lunch":     ["Rice Dal","Rajma Chawal","Veg Biryani","Chole Bhature",
                  "Paneer Butter Masala","Mix Veg Roti","Sambar Rice",
                  "Dal Khichdi","Lemon Rice","Veg Pulao"],
    "Dinner":    ["Roti Sabzi","Dal Makhani","Aloo Matar","Veg Fried Rice",
                  "Paneer Bhurji","Chana Masala","Mixed Dal","Jeera Rice Kadhi",
                  "Veg Noodles","Methi Thepla"],
}

CATEGORIES = {
    "Idli Sambar":"South Indian","Poha":"Snacks","Upma":"South Indian",
    "Bread Omelette":"Continental","Paratha":"North Indian",
    "Cornflakes Milk":"Continental","Aloo Puri":"North Indian",
    "Dosa Chutney":"South Indian","Rice Dal":"Staples","Rajma Chawal":"North Indian",
    "Veg Biryani":"Rice Dishes","Chole Bhature":"North Indian",
    "Paneer Butter Masala":"Paneer","Mix Veg Roti":"Staples",
    "Sambar Rice":"South Indian","Dal Khichdi":"Staples","Lemon Rice":"South Indian",
    "Veg Pulao":"Rice Dishes","Roti Sabzi":"Staples","Dal Makhani":"Dal",
    "Aloo Matar":"North Indian","Veg Fried Rice":"Rice Dishes","Paneer Bhurji":"Paneer",
    "Chana Masala":"North Indian","Mixed Dal":"Dal","Jeera Rice Kadhi":"Staples",
    "Veg Noodles":"Continental","Methi Thepla":"North Indian",
}


def render():
    st.title("🗄️ Data Management")
    st.markdown("> View, add and download mess records. Database backed by **SQLite**.")
    st.markdown("---")

    initialize_database()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 Historical Data", "➕ Add Record",
        "🔮 Prediction History", "📥 Download"
    ])

    # ── Tab 1: View historical dataset ───────────────────────
    with tab1:
        st.subheader("Historical Consumption Records (Database)")
        db_df = fetch_consumption(limit=500)
        if db_df.empty:
            st.info("No records in database yet. Add some via the 'Add Record' tab, "
                    "or run setup.py to bulk-import the synthetic dataset.")
        else:
            st.dataframe(db_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Full Synthetic Dataset (CSV)")
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "mess_dataset.csv"
        )
        if os.path.exists(csv_path):
            csv_df = pd.read_csv(csv_path)
            st.markdown(f"**{len(csv_df):,} records** | Columns: {len(csv_df.columns)}")

            search_col = st.selectbox("Filter by", ["(no filter)","Meal_Period","Food_Item","Weather"])
            if search_col != "(no filter)":
                vals = csv_df[search_col].unique().tolist()
                chosen = st.multiselect(f"Select {search_col}", vals, default=vals[:3])
                csv_df = csv_df[csv_df[search_col].isin(chosen)]

            st.dataframe(csv_df.head(500), use_container_width=True, hide_index=True)
        else:
            st.warning("mess_dataset.csv not found. Run setup.py.")

    # ── Tab 2: Add a new record ───────────────────────────────
    with tab2:
        st.subheader("Add New Consumption Record")
        with st.form("add_record_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                rec_date   = st.date_input("Date", value=date.today())
                meal_period= st.selectbox("Meal Period", ["Breakfast","Lunch","Dinner"])
                weather    = st.selectbox("Weather", ["Sunny","Cloudy","Rainy","Cold","Hot"])
            with c2:
                food_item  = st.selectbox("Food Item", FOOD_ITEMS[meal_period])
                attend     = st.number_input("Student Attendance", 50, 1200, 600, 10)
                exam_day   = st.checkbox("Examination Day")
                holiday    = st.checkbox("Holiday")
            with c3:
                qty_prep   = st.number_input("Qty Prepared", 0.0, 5000.0, 300.0, 5.0)
                qty_cons   = st.number_input("Qty Consumed", 0.0, 5000.0, 260.0, 5.0)

            submitted = st.form_submit_button("💾 Save Record", use_container_width=True)

        if submitted:
            leftover  = qty_prep - qty_cons
            waste_pct = round((leftover / qty_prep) * 100, 2) if qty_prep > 0 else 0.0
            weekend   = int(pd.Timestamp(rec_date).dayofweek >= 5)
            category  = CATEGORIES.get(food_item, "Other")

            insert_consumption({
                "date":          str(rec_date),
                "meal_period":   meal_period,
                "food_item":     food_item,
                "food_category": category,
                "student_attend":int(attend),
                "exam_day":      int(exam_day),
                "holiday":       int(holiday),
                "weekend":       weekend,
                "weather":       weather,
                "qty_prepared":  qty_prep,
                "qty_consumed":  qty_cons,
                "leftover":      leftover,
                "waste_pct":     waste_pct,
                "actual_demand": qty_cons,
            })
            st.success(f"✅ Record saved! Waste: {waste_pct:.1f}%")

    # ── Tab 3: Prediction history ─────────────────────────────
    with tab3:
        st.subheader("Prediction History (from ML model)")
        pred_df = fetch_predictions(limit=200)
        if pred_df.empty:
            st.info("No predictions logged yet. Use the Demand Prediction page.")
        else:
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

    # ── Tab 4: Download ───────────────────────────────────────
    with tab4:
        st.subheader("Download Dataset")
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "mess_dataset.csv"
        )
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                data = f.read()
            st.download_button(
                "📥 Download mess_dataset.csv",
                data=data, file_name="mess_dataset.csv",
                mime="text/csv", use_container_width=True,
            )
            st.caption("⚠️ This is a synthetic dataset for academic use only.")
        else:
            st.warning("Dataset file not found.")

        st.markdown("---")
        db_df2 = fetch_consumption(limit=10000)
        if not db_df2.empty:
            csv_bytes = db_df2.to_csv(index=False).encode()
            st.download_button(
                "📥 Download Database Records (CSV)",
                data=csv_bytes,
                file_name="mess_db_records.csv",
                mime="text/csv",
                use_container_width=True,
            )


render()


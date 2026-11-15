"""
pages/3_Waste_Analytics.py
--------------------------
Food Waste Analytics page.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.waste_analysis import (
    compute_waste_summary, most_wasted_items,
    meal_wise_waste, waste_trend_monthly,
    category_waste, weekly_waste_pattern,
)

st.set_page_config(page_title="Waste Analytics", page_icon="🗑️", layout="wide")

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


@st.cache_data(ttl=300)
def load_data():
    csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "mess_dataset.csv")
    if not os.path.exists(csv):
        return None
    return pd.read_csv(csv, parse_dates=["Date"])


def render():
    st.title("🗑️ Food Waste Analytics")
    st.markdown("> Insight into which meals and items generate the most waste. Data is **synthetic**.")
    st.markdown("---")

    df = load_data()
    if df is None:
        st.error("Dataset not found. Please run setup.py first.")
        return

    # ── KPI row ──────────────────────────────────────────────
    summary = compute_waste_summary(df)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Prepared",   f"{summary['total_prepared']:,}")
    k2.metric("Total Consumed",   f"{summary['total_consumed']:,}")
    k3.metric("Total Wasted",     f"{summary['total_wasted']:,}")
    k4.metric("Avg Waste %",      f"{summary['avg_waste_pct']:.1f}%")
    k5.metric("Est. Savings (₹)", f"₹{summary['est_cost_savings']:,.0f}")

    st.markdown("---")

    # ── Waste trend ───────────────────────────────────────────
    st.subheader("📈 Monthly Waste Trend")
    trend = waste_trend_monthly(df)
    fig_trend = px.line(
        trend, x="YearMonth", y="Avg_Waste_Pct",
        markers=True, labels={"YearMonth":"Month","Avg_Waste_Pct":"Avg Waste %"},
        color_discrete_sequence=["#e53935"],
    )
    fig_trend.add_hline(y=trend["Avg_Waste_Pct"].mean(),
                        line_dash="dot", line_color="#9e9e9e",
                        annotation_text="Overall Avg",
                        annotation_position="bottom right")
    fig_trend.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10,r=10,t=10,b=50))
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Most wasted items & meal-wise ─────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🍛 Top Wasted Food Items")
        top_items = most_wasted_items(df, top_n=12)
        fig_items = px.bar(
            top_items, x="Total_Wasted", y="Food_Item",
            orientation="h", color="Avg_Waste_Pct",
            color_continuous_scale="Reds",
            labels={"Total_Wasted":"Total Wasted", "Food_Item":"Item",
                    "Avg_Waste_Pct":"Avg Waste %"},
        )
        fig_items.update_layout(
            height=420, yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=10),
            coloraxis_showscale=True,
        )
        st.plotly_chart(fig_items, use_container_width=True)

    with col2:
        st.subheader("🍽️ Meal-wise Waste Analysis")
        meal_w = meal_wise_waste(df)
        fig_meal = go.Figure()
        fig_meal.add_trace(go.Bar(
            x=meal_w["Meal_Period"], y=meal_w["Total_Wasted"],
            name="Total Wasted", marker_color="#ef5350",
        ))
        fig_meal.add_trace(go.Bar(
            x=meal_w["Meal_Period"], y=meal_w["Total_Prepared"],
            name="Total Prepared", marker_color="#42a5f5",
        ))
        fig_meal.update_layout(
            barmode="group", height=420,
            yaxis_title="Quantity (units)",
            legend=dict(orientation="h", y=1.1),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=30,b=10),
        )
        st.plotly_chart(fig_meal, use_container_width=True)

    # ── Category breakdown & weekly pattern ──────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📦 Waste by Food Category")
        cat_w = category_waste(df)
        fig_cat = px.treemap(
            cat_w, path=["Food_Category"], values="Total_Wasted",
            color="Avg_Waste_Pct", color_continuous_scale="Reds",
            labels={"Total_Wasted":"Total Wasted","Avg_Waste_Pct":"Avg Waste %"},
        )
        fig_cat.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col4:
        st.subheader("📅 Weekly Waste Pattern")
        week_w = weekly_waste_pattern(df)
        fig_week = px.bar(
            week_w, x="Day_of_Week", y="Avg_Waste_Pct",
            color="Avg_Waste_Pct", color_continuous_scale="Oranges",
            labels={"Day_of_Week":"Day","Avg_Waste_Pct":"Avg Waste %"},
        )
        fig_week.update_layout(
            height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            margin=dict(l=10,r=10,t=10,b=50),
        )
        st.plotly_chart(fig_week, use_container_width=True)

    # ── Waste distribution ────────────────────────────────────
    st.subheader("📊 Waste Percentage Distribution")
    fig_hist = px.histogram(
        df, x="Food_Waste_Percentage", nbins=50,
        color_discrete_sequence=["#ef5350"],
        labels={"Food_Waste_Percentage":"Waste %"},
    )
    fig_hist.update_layout(
        height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=10,b=50),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


render()


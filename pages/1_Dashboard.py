"""
pages/1_Dashboard.py
--------------------
Overview Dashboard page for the Mess Prediction System.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Overview Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
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
h1 {color: #60a5fa !important;} h2 {color: #93c5fd !important;} h3 {color: #bfdbfe !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_data():
    csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "mess_dataset.csv")
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv, parse_dates=["Date"])
    return df


def render():
    st.title("📊 Overview Dashboard")
    st.markdown("> **Synthetic data** for academic demonstration only.")
    st.markdown("---")

    df = load_data()
    if df is None:
        st.error("Dataset not found. Please run the setup script first.")
        st.code("cd mess_prediction_system\npython setup.py")
        return

    # ── KPI row ──────────────────────────────────────────────
    total_meals     = len(df)
    total_prepared  = int(df["Quantity_Prepared"].sum())
    total_wasted    = int(df["Leftover_Quantity"].sum())
    avg_waste_pct   = df["Food_Waste_Percentage"].mean()
    total_consumed  = int(df["Quantity_Consumed"].sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Records",       f"{total_meals:,}")
    k2.metric("Total Prepared (u.)", f"{total_prepared:,}")
    k3.metric("Total Consumed (u.)", f"{total_consumed:,}")
    k4.metric("Total Wasted (u.)",   f"{total_wasted:,}")
    k5.metric("Avg Waste %",         f"{avg_waste_pct:.1f}%")

    st.markdown("---")

    # ── Recent demand trend ───────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Monthly Demand Trend")
        monthly = (
            df.assign(YearMonth=df["Date"].dt.to_period("M").astype(str))
            .groupby("YearMonth")
            .agg(Avg_Demand=("Actual_Food_Demand","mean"),
                 Total_Prepared=("Quantity_Prepared","sum"),
                 Avg_Waste_Pct=("Food_Waste_Percentage","mean"))
            .reset_index()
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["YearMonth"], y=monthly["Avg_Demand"],
            name="Avg Demand", mode="lines+markers",
            line=dict(color="#3949ab", width=2.5),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Bar(
            x=monthly["YearMonth"], y=monthly["Avg_Waste_Pct"],
            name="Waste %", yaxis="y2",
            marker_color="rgba(229,57,53,.35)",
            opacity=0.6,
        ))
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right", title="Waste %"),
            yaxis=dict(title="Avg Demand (units)"),
            legend=dict(orientation="h", y=1.12),
            height=380, margin=dict(l=10,r=10,t=30,b=50),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🍽️ Meal-wise Distribution")
        meal_agg = (
            df.groupby("Meal_Period")
            .agg(Avg_Demand=("Actual_Food_Demand","mean"))
            .reset_index()
        )
        fig2 = px.pie(
            meal_agg, values="Avg_Demand", names="Meal_Period",
            color_discrete_sequence=["#3949ab","#1e88e5","#43a047"],
            hole=0.4,
        )
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10),
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Meal stats table ──────────────────────────────────────
    st.subheader("📋 Meal-wise Statistics")
    meal_stats = (
        df.groupby("Meal_Period")
        .agg(
            Records           = ("Actual_Food_Demand","count"),
            Avg_Demand        = ("Actual_Food_Demand","mean"),
            Total_Prepared    = ("Quantity_Prepared","sum"),
            Total_Consumed    = ("Quantity_Consumed","sum"),
            Total_Wasted      = ("Leftover_Quantity","sum"),
            Avg_Waste_Pct     = ("Food_Waste_Percentage","mean"),
        )
        .round(2)
        .reset_index()
    )
    st.dataframe(meal_stats, use_container_width=True, hide_index=True)

    # ── Weather effect ────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🌦️ Weather vs Avg Demand")
        weather_agg = (
            df.groupby("Weather")
            .agg(Avg_Demand=("Actual_Food_Demand","mean"))
            .reset_index()
            .sort_values("Avg_Demand", ascending=False)
        )
        fig3 = px.bar(weather_agg, x="Weather", y="Avg_Demand",
                      color="Avg_Demand",
                      color_continuous_scale="Blues",
                      labels={"Avg_Demand":"Avg Demand"})
        fig3.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=40),
                           coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("📅 Day-of-Week Demand Pattern")
        day_order = ["Monday","Tuesday","Wednesday","Thursday",
                     "Friday","Saturday","Sunday"]
        day_agg = (
            df.groupby("Day_of_Week")
            .agg(Avg_Demand=("Actual_Food_Demand","mean"))
            .reindex(day_order)
            .reset_index()
        )
        fig4 = px.bar(day_agg, x="Day_of_Week", y="Avg_Demand",
                      color="Avg_Demand",
                      color_continuous_scale="Purples",
                      labels={"Avg_Demand":"Avg Demand","Day_of_Week":"Day"})
        fig4.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=50),
                           coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)


render()


"""
pages/6_Reports.py
------------------
Reports page — generate daily / weekly / monthly analytical summaries.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import io

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")

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


def daily_report(df: pd.DataFrame, report_date: date) -> pd.DataFrame:
    sub = df[df["Date"].dt.date == report_date]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["Meal_Period","Food_Item"])
        .agg(
            Qty_Prepared   = ("Quantity_Prepared",    "sum"),
            Qty_Consumed   = ("Quantity_Consumed",    "sum"),
            Leftover       = ("Leftover_Quantity",    "sum"),
            Avg_Waste_Pct  = ("Food_Waste_Percentage","mean"),
            Actual_Demand  = ("Actual_Food_Demand",   "mean"),
        )
        .round(2)
        .reset_index()
    )


def weekly_report(df: pd.DataFrame, week_start: date) -> pd.DataFrame:
    week_end = week_start + timedelta(days=6)
    sub = df[(df["Date"].dt.date >= week_start) & (df["Date"].dt.date <= week_end)]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["Meal_Period","Food_Item"])
        .agg(
            Total_Prepared = ("Quantity_Prepared","sum"),
            Total_Consumed = ("Quantity_Consumed","sum"),
            Total_Wasted   = ("Leftover_Quantity","sum"),
            Avg_Waste_Pct  = ("Food_Waste_Percentage","mean"),
        )
        .round(2)
        .reset_index()
    )


def monthly_report(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    sub = df[(df["Date"].dt.year == year) & (df["Date"].dt.month == month)]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["Meal_Period","Food_Item"])
        .agg(
            Total_Prepared = ("Quantity_Prepared","sum"),
            Total_Consumed = ("Quantity_Consumed","sum"),
            Total_Wasted   = ("Leftover_Quantity","sum"),
            Avg_Waste_Pct  = ("Food_Waste_Percentage","mean"),
            Total_Demand   = ("Actual_Food_Demand","sum"),
        )
        .round(2)
        .reset_index()
    )


def render():
    st.title("📄 Reports")
    st.markdown("> Generate and download analytical reports from the synthetic dataset.")
    st.markdown("---")

    df = load_data()
    if df is None:
        st.error("Dataset not found. Run setup.py first.")
        return

    tab1, tab2, tab3 = st.tabs(["📅 Daily Report", "📆 Weekly Report", "🗓️ Monthly Report"])

    # ── Daily ─────────────────────────────────────────────────
    with tab1:
        st.subheader("Daily Report")
        min_date = df["Date"].dt.date.min()
        max_date = df["Date"].dt.date.max()
        sel_date = st.date_input("Select date", value=max_date,
                                 min_value=min_date, max_value=max_date)
        rpt = daily_report(df, sel_date)

        if rpt.empty:
            st.warning("No data found for the selected date.")
        else:
            st.markdown(f"### Report for **{sel_date}**")
            # Summary KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Prepared", f"{rpt['Qty_Prepared'].sum():,.0f}")
            k2.metric("Total Consumed", f"{rpt['Qty_Consumed'].sum():,.0f}")
            k3.metric("Total Leftover", f"{rpt['Leftover'].sum():,.0f}")
            k4.metric("Avg Waste %", f"{rpt['Avg_Waste_Pct'].mean():.1f}%")

            st.dataframe(rpt, use_container_width=True, hide_index=True)

            fig = px.bar(rpt, x="Food_Item", y=["Qty_Prepared","Qty_Consumed","Leftover"],
                         barmode="group", facet_col="Meal_Period",
                         labels={"value":"Quantity","variable":"Metric"},
                         color_discrete_sequence=["#42a5f5","#66bb6a","#ef5350"])
            fig.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10,r=10,t=40,b=80))
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            csv = rpt.to_csv(index=False).encode()
            st.download_button("📥 Download Daily Report", csv,
                               f"daily_report_{sel_date}.csv", "text/csv")

    # ── Weekly ────────────────────────────────────────────────
    with tab2:
        st.subheader("Weekly Report")
        default_week = max_date - timedelta(days=max_date.weekday())
        week_start = st.date_input("Week starting (Monday)", value=default_week,
                                   min_value=min_date, max_value=max_date)
        week_end   = week_start + timedelta(days=6)
        st.caption(f"Week: {week_start} → {week_end}")
        rpt_w = weekly_report(df, week_start)

        if rpt_w.empty:
            st.warning("No data found for the selected week.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Prepared", f"{rpt_w['Total_Prepared'].sum():,.0f}")
            k2.metric("Total Consumed", f"{rpt_w['Total_Consumed'].sum():,.0f}")
            k3.metric("Total Wasted",   f"{rpt_w['Total_Wasted'].sum():,.0f}")
            k4.metric("Avg Waste %",    f"{rpt_w['Avg_Waste_Pct'].mean():.1f}%")

            st.dataframe(rpt_w, use_container_width=True, hide_index=True)

            fig_w = px.bar(rpt_w, x="Food_Item", y="Total_Wasted",
                           color="Meal_Period",
                           labels={"Total_Wasted":"Wasted Units"},
                           color_discrete_sequence=["#ef5350","#ff7043","#ffa726"])
            fig_w.update_layout(height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=10,r=10,t=10,b=80))
            fig_w.update_xaxes(tickangle=45)
            st.plotly_chart(fig_w, use_container_width=True)

            csv = rpt_w.to_csv(index=False).encode()
            st.download_button("📥 Download Weekly Report", csv,
                               f"weekly_report_{week_start}.csv", "text/csv")

    # ── Monthly ───────────────────────────────────────────────
    with tab3:
        st.subheader("Monthly Report")
        c1, c2 = st.columns(2)
        years  = sorted(df["Date"].dt.year.unique(), reverse=True)
        sel_year  = c1.selectbox("Year",  years)
        months_in_year = sorted(
            df[df["Date"].dt.year == sel_year]["Date"].dt.month.unique()
        )
        sel_month = c2.selectbox("Month", months_in_year,
                                 format_func=lambda m: date(2000, m, 1).strftime("%B"))

        rpt_m = monthly_report(df, sel_year, sel_month)

        if rpt_m.empty:
            st.warning("No data found for the selected month.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Prepared", f"{rpt_m['Total_Prepared'].sum():,.0f}")
            k2.metric("Total Consumed", f"{rpt_m['Total_Consumed'].sum():,.0f}")
            k3.metric("Total Wasted",   f"{rpt_m['Total_Wasted'].sum():,.0f}")
            k4.metric("Avg Waste %",    f"{rpt_m['Avg_Waste_Pct'].mean():.1f}%")

            st.dataframe(rpt_m, use_container_width=True, hide_index=True)

            fig_m = px.pie(rpt_m, names="Meal_Period", values="Total_Demand",
                           title=f"Demand Share — {date(sel_year, sel_month, 1).strftime('%B %Y')}",
                           hole=0.4,
                           color_discrete_sequence=["#3949ab","#1e88e5","#43a047"])
            fig_m.update_layout(height=380, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig_m, use_container_width=True)

            csv = rpt_m.to_csv(index=False).encode()
            mon_name = date(sel_year, sel_month, 1).strftime("%B_%Y")
            st.download_button("📥 Download Monthly Report", csv,
                               f"monthly_report_{mon_name}.csv", "text/csv")


render()


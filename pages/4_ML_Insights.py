"""
pages/4_ML_Insights.py
-----------------------
Machine Learning Insights page — model comparison, feature importance, residuals.
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.predict   import load_model, load_metadata
from src.evaluate  import get_feature_importances, residual_stats
from src.preprocessing import prepare_features

st.set_page_config(page_title="ML Insights", page_icon="🧠", layout="wide")

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


@st.cache_data(ttl=600)
def load_data_and_meta():
    csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "mess_dataset.csv")
    if not os.path.exists(csv):
        return None, {}
    df   = pd.read_csv(csv, parse_dates=["Date"])
    meta = load_metadata()
    return df, meta


def render():
    st.title("🧠 Machine Learning Insights")
    st.markdown("> Transparent model evaluation and feature analysis. No metrics are fabricated.")
    st.markdown("---")

    df, meta = load_data_and_meta()
    if df is None:
        st.error("Dataset not found. Run setup.py first.")
        return
    if not meta:
        st.error("Model metadata not found. Run setup.py to train models.")
        return

    # ── Best model banner ─────────────────────────────────────
    best_name = meta.get("best_model_name", "Unknown")
    st.success(f"🏆 **Best Model Selected:** {best_name}")

    # ── Algorithm comparison ──────────────────────────────────
    st.subheader("📊 Algorithm Comparison")
    model_results = meta.get("model_results", {})
    rows = []
    for name, info in model_results.items():
        vm = info.get("val_metrics", {})
        tm = info.get("test_metrics", {})
        rows.append({
            "Model":        name,
            "Val MAE":      vm.get("MAE", "-"),
            "Val RMSE":     vm.get("RMSE", "-"),
            "Val R²":       vm.get("R2", "-"),
            "Test MAE":     tm.get("MAE", "-"),
            "Test RMSE":    tm.get("RMSE", "-"),
            "Test R²":      tm.get("R2", "-"),
            "Train Time(s)":info.get("train_time_s", "-"),
        })
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── Metric comparison charts ───────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig_mae = px.bar(
            comp_df, x="Model", y="Val MAE",
            color="Model", text="Val MAE",
            title="Validation MAE (lower is better)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_mae.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_mae.update_layout(showlegend=False, height=350,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_mae, use_container_width=True)

    with col2:
        fig_r2 = px.bar(
            comp_df, x="Model", y="Val R²",
            color="Model", text="Val R²",
            title="Validation R² Score (higher is better)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_r2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_r2.update_layout(showlegend=False, height=350,
                             yaxis=dict(range=[0, 1.05]),
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_r2, use_container_width=True)

    # ── Feature importance ────────────────────────────────────
    st.subheader("🔍 Feature Importance (Best Model)")
    try:
        pipeline = load_model()
        X_sample, y_sample = prepare_features(df.tail(2000))
        fi_df = get_feature_importances(pipeline, X_sample.columns.tolist())

        if not fi_df.empty:
            top_fi = fi_df.head(20)
            fig_fi = px.bar(
                top_fi, x="Importance", y="Feature",
                orientation="h", color="Importance",
                color_continuous_scale="Blues",
                labels={"Importance":"Importance Score"},
            )
            fig_fi.update_layout(
                height=550, yaxis=dict(autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin=dict(l=10,r=10,t=10,b=10),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type (Linear Regression).")

        # ── Actual vs Predicted ───────────────────────────────
        st.subheader("🎯 Actual vs Predicted (Test Sample)")
        n_sample = min(500, len(df))
        test_sample = df.tail(n_sample).copy()
        X_t, y_t = prepare_features(test_sample)
        y_pred   = pipeline.predict(X_t)

        avp_df = pd.DataFrame({
            "Actual":    y_t.values,
            "Predicted": y_pred,
        })
        fig_avp = px.scatter(
            avp_df, x="Actual", y="Predicted",
            opacity=0.5, color_discrete_sequence=["#3949ab"],
            labels={"Actual":"Actual Demand","Predicted":"Predicted Demand"},
        )
        # Perfect prediction line
        min_v = float(avp_df.min().min())
        max_v = float(avp_df.max().max())
        fig_avp.add_shape(type="line", x0=min_v, y0=min_v,
                          x1=max_v, y1=max_v,
                          line=dict(color="#e53935", width=2, dash="dash"))
        fig_avp.update_layout(
            height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=10),
        )
        st.plotly_chart(fig_avp, use_container_width=True)

        # ── Residual analysis ─────────────────────────────────
        st.subheader("📉 Residual Analysis")
        res = residual_stats(y_t.values, y_pred)
        col3, col4, col5 = st.columns(3)
        col3.metric("Residual Mean", f"{res['mean']:.3f}")
        col4.metric("Residual Std",  f"{res['std']:.3f}")
        col5.metric("Residual Median", f"{res['median']:.3f}")

        fig_res = px.histogram(
            x=res["residuals"], nbins=60,
            labels={"x":"Residual (Actual − Predicted)"},
            color_discrete_sequence=["#5c6bc0"],
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="#e53935",
                          annotation_text="Zero error",
                          annotation_position="top right")
        fig_res.update_layout(
            height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=50),
        )
        st.plotly_chart(fig_res, use_container_width=True)

    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run setup.py to train and save the model.")

    # ── Training dataset stats ────────────────────────────────
    st.subheader("📋 Training Dataset Statistics")
    st.markdown(f"- **Total records:** {len(df):,}")
    st.markdown(f"- **Date range:** {df['Date'].min().date()} → {df['Date'].max().date()}")
    st.markdown(f"- **Meal periods:** {', '.join(df['Meal_Period'].unique())}")
    st.markdown(f"- **Unique food items:** {df['Food_Item'].nunique()}")
    st.markdown(f"- **Train features used:** {meta.get('train_features', [])}")

    with st.expander("📊 Full Descriptive Statistics"):
        num_cols = ["Actual_Food_Demand","Quantity_Prepared","Quantity_Consumed",
                    "Leftover_Quantity","Food_Waste_Percentage","Student_Attendance"]
        st.dataframe(df[num_cols].describe().round(2), use_container_width=True)


render()


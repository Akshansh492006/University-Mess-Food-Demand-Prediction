"""
app.py
------
Entry point for the University Mess Food Demand & Waste Prediction System.
Run with: streamlit run app.py

Author : ML Engineering Team
Date   : 2024
"""

import os
import sys

# Ensure src/ is on the Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import streamlit as st

st.set_page_config(
    page_title = "University Mess Prediction System",
    page_icon  = "🍽️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        "About": "University Mess Food Demand & Waste Prediction System — INT395 Academic Project",
    }
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- sidebar ---- */
[data-testid="stSidebar"] {background: linear-gradient(180deg,#1a1f36 0%,#2d3561 100%);}
[data-testid="stSidebar"] * {color:#e8eaf6 !important;}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {color:#90caf9 !important;}

/* ---- metric cards ---- */
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

/* ---- section headers ---- */
h1 {color: #60a5fa !important; font-weight: 800;}
h2 {color: #93c5fd !important; font-weight: 700;}
h3 {color: #bfdbfe !important; font-weight: 600;}

/* ---- dataframe ---- */
[data-testid="stDataFrame"] {border-radius:10px; overflow:hidden;}

/* ---- tabs ---- */
[data-baseweb="tab-list"] {gap:6px;}
[data-baseweb="tab"]      {border-radius:8px 8px 0 0;}

/* ---- buttons ---- */
.stButton>button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 10px 24px !important;
    transition: transform .15s, box-shadow .15s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
}

/* ---- info banner ---- */
.info-banner {
    background: rgba(34, 197, 94, 0.12) !important; border-left: 4px solid #22c55e !important;
    border-radius: 8px !important; padding: 12px 16px !important; margin-bottom: 16px !important;
    font-size: 0.9rem !important; color: #86efac !important;
}
</style>
""", unsafe_allow_html=True)


# ── Synthetic data notice ─────────────────────────────────────
st.markdown("""
<div class="info-banner">
⚠️ <strong>Academic Project — Synthetic Dataset:</strong>
All data displayed in this system is <em>synthetically generated</em>
for demonstration purposes and does NOT represent real university mess records.
</div>
""", unsafe_allow_html=True)


# ── Home / landing page ───────────────────────────────────────
st.title("🍽️ University Mess Food Demand & Waste Prediction System")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    ### 📊 Overview Dashboard
    View real-time statistics, demand trends and meal-wise summaries.
    Navigate using the **sidebar** on the left.
    """)
with col2:
    st.markdown("""
    ### 🤖 ML-Powered Predictions
    Our **Random Forest** model predicts food demand so the mess team can
    prepare the right quantity and minimise waste.
    """)
with col3:
    st.markdown("""
    ### 🗑️ Waste Analytics
    Identify which meals and items waste the most food, and track
    improvement over time.
    """)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9e9e9e; font-size:.82rem; margin-top:24px;">
University Mess Food Demand & Waste Prediction System &nbsp;|&nbsp;
INT395 Academic Project &nbsp;|&nbsp; Synthetic Data Only
</div>
""", unsafe_allow_html=True)


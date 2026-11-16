"""
waste_analysis.py
Food waste analytics module for the University Mess Prediction System.
Aggregates waste distributions, item patterns, meal period comparisons,
and financial waste savings estimates.
"""

from typing import Dict, Any
import pandas as pd


def compute_waste_summary(df: pd.DataFrame, unit_cost: float = 30.0) -> Dict[str, Any]:
    """Calculate aggregate quantity prepared, consumed, wasted, and financial impact."""
    waste_col = "Actual_Food_Waste" if "Actual_Food_Waste" in df.columns else "Leftover_Quantity"
    total_prepared = float(df["Quantity_Prepared"].sum())
    total_consumed = float(df["Quantity_Consumed"].sum())
    total_wasted = float(df[waste_col].sum())
    avg_waste_pct = float(df["Food_Waste_Percentage"].mean())

    target_waste = total_prepared * 0.05
    excess_waste = max(total_wasted - target_waste, 0.0)
    cost_savings = excess_waste * unit_cost

    return {
        "total_prepared": int(total_prepared),
        "total_consumed": int(total_consumed),
        "total_wasted": int(total_wasted),
        "avg_waste_pct": round(avg_waste_pct, 2),
        "est_cost_savings": round(cost_savings, 2),
    }


def most_wasted_items(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Identify top dishes by total leftover volume and average wastage percentage."""
    waste_col = "Actual_Food_Waste" if "Actual_Food_Waste" in df.columns else "Leftover_Quantity"
    return (
        df.groupby("Food_Item")
        .agg(
            Total_Wasted=(waste_col, "sum"),
            Avg_Waste_Pct=("Food_Waste_Percentage", "mean"),
        )
        .reset_index()
        .sort_values("Total_Wasted", ascending=False)
        .head(top_n)
    )


def most_demanded_items(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Identify highest demand dishes across all recorded meal periods."""
    demand_col = "Actual_Food_Demand" if "Actual_Food_Demand" in df.columns else "Quantity_Consumed"
    return (
        df.groupby(["Food_Item", "Meal_Period"])
        .agg(
            Total_Demand=(demand_col, "sum"),
            Avg_Demand=(demand_col, "mean"),
        )
        .reset_index()
        .sort_values("Total_Demand", ascending=False)
        .head(top_n)
    )


def meal_wise_waste(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate waste and preparation volumes partitioned by meal period."""
    waste_col = "Actual_Food_Waste" if "Actual_Food_Waste" in df.columns else "Leftover_Quantity"
    grouped = (
        df.groupby("Meal_Period")
        .agg(
            Total_Prepared=("Quantity_Prepared", "sum"),
            Total_Wasted=(waste_col, "sum"),
            Avg_Waste_Pct=("Food_Waste_Percentage", "mean"),
        )
        .reset_index()
    )
    grouped["Waste_Fraction"] = grouped["Total_Wasted"] / grouped["Total_Prepared"]
    return grouped


def waste_trend_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate month-by-month temporal waste rate trends."""
    waste_col = "Actual_Food_Waste" if "Actual_Food_Waste" in df.columns else "Leftover_Quantity"
    working = df.copy()
    working["Date"] = pd.to_datetime(working["Date"])
    working["YearMonth"] = working["Date"].dt.to_period("M").astype(str)

    return (
        working.groupby("YearMonth")
        .agg(
            Avg_Waste_Pct=("Food_Waste_Percentage", "mean"),
            Total_Wasted=(waste_col, "sum"),
            Total_Prepared=("Quantity_Prepared", "sum"),
        )
        .reset_index()
    )


def category_waste(df: pd.DataFrame) -> pd.DataFrame:
    """Compute waste breakdown segmented by food cuisine category."""
    waste_col = "Actual_Food_Waste" if "Actual_Food_Waste" in df.columns else "Leftover_Quantity"
    return (
        df.groupby("Food_Category")
        .agg(
            Total_Wasted=(waste_col, "sum"),
            Avg_Waste_Pct=("Food_Waste_Percentage", "mean"),
        )
        .reset_index()
        .sort_values("Total_Wasted", ascending=False)
    )


def weekly_waste_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean food waste percentage per day of the week."""
    order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    return (
        df.groupby("Day_of_Week")
        .agg(Avg_Waste_Pct=("Food_Waste_Percentage", "mean"))
        .reindex(order)
        .reset_index()
    )

"""
waste_analysis.py
-----------------
Food waste analytics module for the Mess Prediction System.

Author : ML Engineering Team
Date   : 2024
"""

import pandas as pd
import numpy as np


def compute_waste_summary(df: pd.DataFrame) -> dict:
    """
    Compute high-level waste summary statistics from the dataset.

    Parameters
    ----------
    df : DataFrame with columns Quantity_Prepared, Quantity_Consumed,
         Leftover_Quantity, Food_Waste_Percentage, etc.

    Returns
    -------
    dict with summary metrics
    """
    total_prepared = df["Quantity_Prepared"].sum()
    total_consumed = df["Quantity_Consumed"].sum()
    total_wasted   = df["Leftover_Quantity"].sum()
    avg_waste_pct  = df["Food_Waste_Percentage"].mean()

    # Estimated cost savings if waste were reduced to 5 %
    # (assume ₹5 per unit as a placeholder)
    unit_cost       = 5
    target_waste    = total_prepared * 0.05
    excess_waste    = max(total_wasted - target_waste, 0)
    cost_savings    = excess_waste * unit_cost

    return {
        "total_prepared":    int(total_prepared),
        "total_consumed":    int(total_consumed),
        "total_wasted":      int(total_wasted),
        "avg_waste_pct":     round(avg_waste_pct, 2),
        "est_cost_savings":  round(cost_savings, 2),
    }


def most_wasted_items(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Return the top N food items by total leftover quantity.

    Returns
    -------
    DataFrame with columns: Food_Item, Total_Wasted, Avg_Waste_Pct
    """
    grouped = (
        df.groupby("Food_Item")
        .agg(
            Total_Wasted   = ("Leftover_Quantity",    "sum"),
            Avg_Waste_Pct  = ("Food_Waste_Percentage","mean"),
        )
        .reset_index()
        .sort_values("Total_Wasted", ascending=False)
        .head(top_n)
    )
    return grouped


def meal_wise_waste(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate waste statistics by Meal_Period.

    Returns
    -------
    DataFrame with columns: Meal_Period, Total_Prepared, Total_Wasted,
                            Avg_Waste_Pct, Waste_Fraction
    """
    grouped = (
        df.groupby("Meal_Period")
        .agg(
            Total_Prepared = ("Quantity_Prepared",    "sum"),
            Total_Wasted   = ("Leftover_Quantity",    "sum"),
            Avg_Waste_Pct  = ("Food_Waste_Percentage","mean"),
        )
        .reset_index()
    )
    grouped["Waste_Fraction"] = grouped["Total_Wasted"] / grouped["Total_Prepared"]
    return grouped


def waste_trend_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly waste percentage trend.

    Requires a 'Date' column.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    trend = (
        df.groupby("YearMonth")
        .agg(
            Avg_Waste_Pct  = ("Food_Waste_Percentage","mean"),
            Total_Wasted   = ("Leftover_Quantity",    "sum"),
            Total_Prepared = ("Quantity_Prepared",    "sum"),
        )
        .reset_index()
    )
    return trend


def category_waste(df: pd.DataFrame) -> pd.DataFrame:
    """Waste breakdown by Food_Category."""
    grouped = (
        df.groupby("Food_Category")
        .agg(
            Total_Wasted  = ("Leftover_Quantity",    "sum"),
            Avg_Waste_Pct = ("Food_Waste_Percentage","mean"),
        )
        .reset_index()
        .sort_values("Total_Wasted", ascending=False)
    )
    return grouped


def weekly_waste_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Average waste percentage by Day_of_Week."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
    grouped = (
        df.groupby("Day_of_Week")
        .agg(Avg_Waste_Pct=("Food_Waste_Percentage","mean"))
        .reindex(order)
        .reset_index()
    )
    return grouped


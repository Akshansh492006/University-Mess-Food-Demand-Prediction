"""
tests/test_system.py
--------------------
Unit and integration tests for the Mess Prediction System.

Run with:
    python -m pytest tests/ -v

Author : ML Engineering Team
Date   : 2024
"""

import os
import sys
import json
import shutil
import tempfile

import pytest
import numpy as np
import pandas as pd

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_generation import generate_dataset
from src.preprocessing   import (
    clean_dataframe, build_preprocessor,
    prepare_features, chronological_split
)
from src.evaluate        import evaluate_model, get_feature_importances, residual_stats
from src.waste_analysis  import (
    compute_waste_summary, most_wasted_items,
    meal_wise_waste, waste_trend_monthly
)
from src.database        import initialize_database, insert_prediction, fetch_predictions

from sklearn.dummy        import DummyRegressor
from sklearn.pipeline     import Pipeline


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def small_df():
    """Generate a small dataset for fast tests."""
    from datetime import date
    return generate_dataset(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
    )


@pytest.fixture(scope="session")
def splits(small_df):
    df = clean_dataframe(small_df)
    return chronological_split(df)


@pytest.fixture(scope="session")
def fitted_pipeline(splits):
    """Fit a DummyRegressor pipeline on training split (fast)."""
    train_df, _, _ = splits
    X_train, y_train = prepare_features(train_df)

    pre      = build_preprocessor()
    pipeline = Pipeline([("preprocessor", pre), ("model", DummyRegressor())])
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, y_train


# ── Tests: data_generation ───────────────────────────────────

class TestDataGeneration:
    def test_shape(self, small_df):
        """Dataset should have more than 1 row and expected columns."""
        assert len(small_df) > 0
        required = ["Date","Meal_Period","Food_Item","Actual_Food_Demand",
                    "Quantity_Prepared","Quantity_Consumed","Leftover_Quantity",
                    "Food_Waste_Percentage","Student_Attendance"]
        for col in required:
            assert col in small_df.columns, f"Missing column: {col}"

    def test_demand_positive(self, small_df):
        assert (small_df["Actual_Food_Demand"] >= 0).all()

    def test_waste_pct_range(self, small_df):
        assert small_df["Food_Waste_Percentage"].between(0, 100).all()

    def test_no_all_nan(self, small_df):
        assert not small_df.isnull().all(axis=0).any()

    def test_meal_periods(self, small_df):
        assert set(small_df["Meal_Period"].unique()) == {"Breakfast","Lunch","Dinner"}

    def test_save_csv(self, tmp_path, small_df):
        """Dataset can be saved to CSV."""
        from datetime import date
        out = str(tmp_path / "test_dataset.csv")
        df  = generate_dataset(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            output_path=out,
        )
        assert os.path.exists(out)
        df2 = pd.read_csv(out)
        assert len(df2) > 0


# ── Tests: preprocessing ─────────────────────────────────────

class TestPreprocessing:
    def test_clean_no_duplicates(self, small_df):
        df = small_df.copy()
        df = pd.concat([df, df.head(10)], ignore_index=True)
        cleaned = clean_dataframe(df)
        assert cleaned.duplicated().sum() == 0

    def test_prepare_features_no_target_in_X(self, splits):
        train_df, _, _ = splits
        X, y = prepare_features(train_df)
        assert "Actual_Food_Demand" not in X.columns

    def test_prepare_features_no_leaky_cols(self, splits):
        train_df, _, _ = splits
        X, _ = prepare_features(train_df)
        leaky = ["Quantity_Prepared","Quantity_Consumed",
                 "Leftover_Quantity","Food_Waste_Percentage"]
        for col in leaky:
            assert col not in X.columns

    def test_chronological_split_order(self, small_df):
        df = clean_dataframe(small_df)
        tr, va, te = chronological_split(df)
        assert tr["Date"].max() <= va["Date"].min()
        assert va["Date"].max() <= te["Date"].min()

    def test_preprocessor_transforms(self, splits):
        train_df, _, _ = splits
        X, y = prepare_features(train_df)
        pre = build_preprocessor()
        Xt  = pre.fit_transform(X, y)
        assert Xt.shape[0] == len(X)
        assert Xt.shape[1] > X.shape[1]  # one-hot expands features


# ── Tests: evaluate ──────────────────────────────────────────

class TestEvaluate:
    def test_metrics_keys(self, fitted_pipeline):
        pipeline, X, y = fitted_pipeline
        metrics = evaluate_model(pipeline, X, y)
        for key in ["MAE","MSE","RMSE","R2","predictions"]:
            assert key in metrics

    def test_metrics_non_negative(self, fitted_pipeline):
        pipeline, X, y = fitted_pipeline
        m = evaluate_model(pipeline, X, y)
        assert m["MAE"]  >= 0
        assert m["MSE"]  >= 0
        assert m["RMSE"] >= 0

    def test_residual_stats(self, fitted_pipeline):
        pipeline, X, y = fitted_pipeline
        y_pred = pipeline.predict(X)
        stats  = residual_stats(y.values, y_pred)
        assert "residuals" in stats
        assert len(stats["residuals"]) == len(y)


# ── Tests: waste_analysis ────────────────────────────────────

class TestWasteAnalysis:
    def test_summary_keys(self, small_df):
        summary = compute_waste_summary(small_df)
        for key in ["total_prepared","total_consumed","total_wasted",
                    "avg_waste_pct","est_cost_savings"]:
            assert key in summary

    def test_most_wasted_items_top_n(self, small_df):
        df = most_wasted_items(small_df, top_n=5)
        assert len(df) <= 5
        assert "Total_Wasted" in df.columns

    def test_meal_wise_waste(self, small_df):
        df = meal_wise_waste(small_df)
        assert set(df["Meal_Period"]) == {"Breakfast","Lunch","Dinner"}

    def test_monthly_trend(self, small_df):
        trend = waste_trend_monthly(small_df)
        assert "YearMonth" in trend.columns
        assert "Avg_Waste_Pct" in trend.columns


# ── Tests: database ───────────────────────────────────────────

class TestDatabase:
    def test_initialize_creates_tables(self, tmp_path):
        db = str(tmp_path / "test.db")
        initialize_database(db_path=db)
        import sqlite3
        conn = sqlite3.connect(db)
        tables = [r[0] for r in
                  conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        expected = ["food_items","historical_consumption","attendance_records",
                    "menu_records","prediction_history","waste_records"]
        for t in expected:
            assert t in tables, f"Table '{t}' not created"

    def test_insert_and_fetch_prediction(self, tmp_path):
        db = str(tmp_path / "pred_test.db")
        initialize_database(db_path=db)
        record = {
            "date": "2024-06-01", "meal_period": "Lunch",
            "food_item": "Rice Dal", "food_category": "Staples",
            "student_attendance": 600, "exam_day": 0, "holiday": 0,
            "weather": "Sunny", "predicted_demand": 285.5,
            "suggested_qty": 320, "estimated_surplus": 34.5,
            "model_name": "Random Forest",
        }
        rowid = insert_prediction(record, db_path=db)
        assert rowid >= 1

        df = fetch_predictions(limit=10, db_path=db)
        assert len(df) >= 1
        assert df.iloc[0]["food_item"] == "Rice Dal"


# ── Tests: predict (integration) ─────────────────────────────

class TestPredictionIntegration:
    def test_predict_returns_dict(self):
        """
        Requires that setup.py has been run and best_model.pkl exists.
        Skips if model not found.
        """
        model_path = os.path.join(ROOT, "models", "best_model.pkl")
        if not os.path.exists(model_path):
            pytest.skip("Model not trained yet — run setup.py first.")

        from src.predict import predict_demand
        from datetime import date

        result = predict_demand(
            date_obj              = date(2025, 3, 15),
            meal_period           = "Lunch",
            food_item             = "Veg Biryani",
            food_category         = "Rice Dishes",
            student_attendance    = 650,
            examination_day       = 0,
            holiday               = 0,
            weather               = "Sunny",
            previous_day_demand   = 300.0,
            previous_week_demand  = 295.0,
            historical_avg_demand = 298.0,
            buffer_pct            = 12.0,
        )
        assert "predicted_demand"  in result
        assert "suggested_qty"     in result
        assert "estimated_surplus" in result
        assert result["predicted_demand"] >= 0

    def test_missing_model_raises_error(self):
        """FileNotFoundError is raised when model is missing."""
        from src import predict as pred_module
        pred_module._model_cache = None  # reset cache
        orig_path = pred_module.MODEL_PATH
        pred_module.MODEL_PATH = "/nonexistent/model.pkl"
        with pytest.raises(FileNotFoundError):
            pred_module.load_model()
        pred_module.MODEL_PATH = orig_path
        pred_module._model_cache = None


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v"])


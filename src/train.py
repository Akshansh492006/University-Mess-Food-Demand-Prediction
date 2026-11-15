"""
train.py
--------
Model training, hyperparameter search and model selection.

Author : ML Engineering Team
Date   : 2024
"""

import os
import sys
import json
import time
import joblib
import numpy as np

from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.dummy           import DummyRegressor
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import (
    build_preprocessor, prepare_features, chronological_split, clean_dataframe
)
from src.evaluate import evaluate_model

RANDOM_SEED  = 42
MODELS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _build_pipeline(model) -> Pipeline:
    """Wrap preprocessor + model in a single sklearn Pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model",        model),
    ])


def _param_grid(name: str) -> dict:
    """Return hyperparameter search space for RandomizedSearchCV."""
    if name == "Decision Tree":
        return {
            "model__max_depth":        [4, 6, 8, 10, 12, None],
            "model__min_samples_leaf": [2, 4, 8, 16],
            "model__min_samples_split":[2, 5, 10],
        }
    if name == "Random Forest":
        return {
            "model__n_estimators":     [100, 200, 300],
            "model__max_depth":        [6, 8, 10, None],
            "model__min_samples_leaf": [2, 4, 8],
            "model__max_features":     ["sqrt", "log2", 0.5],
        }
    return {}   # Linear Regression / Dummy — no tuning needed


def train_all_models(train_df, val_df, test_df,
                     n_iter: int = 20) -> dict:
    """
    Train all candidate models on train_df, select best on val_df,
    report final test performance.

    Returns
    -------
    dict with keys: 'results', 'best_model_name', 'best_pipeline'
    """
    X_train, y_train = prepare_features(train_df)
    X_val,   y_val   = prepare_features(val_df)
    X_test,  y_test  = prepare_features(test_df)

    candidate_models = {
        "Dummy Baseline":  DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Decision Tree":   DecisionTreeRegressor(random_state=RANDOM_SEED),
        "Random Forest":   RandomForestRegressor(
                               random_state=RANDOM_SEED, n_jobs=-1),
    }

    results = {}

    for name, base_model in candidate_models.items():
        print(f"\n{'='*55}")
        print(f"  Training: {name}")
        print(f"{'='*55}")
        t0 = time.time()

        pipeline = _build_pipeline(base_model)
        grid     = _param_grid(name)

        if grid:
            # Use TimeSeriesSplit on training data for CV
            tscv = TimeSeriesSplit(n_splits=5)
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=grid,
                n_iter=n_iter,
                cv=tscv,
                scoring="neg_mean_squared_error",
                refit=True,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            print(f"  Best params : {search.best_params_}")
        else:
            pipeline.fit(X_train, y_train)
            best_pipe = pipeline

        elapsed = time.time() - t0

        # Evaluate on val set (model selection)
        val_metrics  = evaluate_model(best_pipe, X_val,  y_val,  label="Val")
        # Evaluate on test set (unbiased final report)
        test_metrics = evaluate_model(best_pipe, X_test, y_test, label="Test")

        results[name] = {
            "pipeline":      best_pipe,
            "val_metrics":   val_metrics,
            "test_metrics":  test_metrics,
            "train_time_s":  round(elapsed, 2),
        }

        print(f"  Val  → MAE={val_metrics['MAE']:.2f}  RMSE={val_metrics['RMSE']:.2f}  R²={val_metrics['R2']:.4f}")
        print(f"  Test → MAE={test_metrics['MAE']:.2f}  RMSE={test_metrics['RMSE']:.2f}  R²={test_metrics['R2']:.4f}")

    # ── Model selection: best validation R² ──────────────────
    # Exclude dummy from "best"
    scored = {k: v for k, v in results.items() if k != "Dummy Baseline"}
    best_name = max(scored, key=lambda k: scored[k]["val_metrics"]["R2"])
    best_pipeline = results[best_name]["pipeline"]

    print(f"\n✅  Best model selected: {best_name}")
    return {
        "results":          results,
        "best_model_name":  best_name,
        "best_pipeline":    best_pipeline,
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }


def save_model(pipeline, model_name: str, results: dict,
               train_shape: tuple, feature_names: list):
    """Save model and metadata to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"[train] Model saved → {model_path}")

    # Build serialisable metrics
    def _clean(d):
        return {k: float(v) for k, v in d.items() if k != "predictions"}

    meta = {
        "best_model_name":  model_name,
        "train_rows":       train_shape[0],
        "train_features":   feature_names,
        "model_results":    {
            name: {
                "val_metrics":  _clean(info["val_metrics"]),
                "test_metrics": _clean(info["test_metrics"]),
                "train_time_s": info["train_time_s"],
            }
            for name, info in results.items()
        },
    }

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Metadata saved → {meta_path}")

    return model_path, meta_path


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_generation import generate_dataset

    base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "mess_dataset.csv")

    if not os.path.exists(csv_path):
        print("[train] Generating dataset …")
        df = generate_dataset(output_path=csv_path)
    else:
        import pandas as pd
        print("[train] Loading existing dataset …")
        df = pd.read_csv(csv_path, parse_dates=["Date"])

    df = clean_dataframe(df)
    train_df, val_df, test_df = chronological_split(df)

    out = train_all_models(train_df, val_df, test_df)

    save_model(
        pipeline      = out["best_pipeline"],
        model_name    = out["best_model_name"],
        results       = out["results"],
        train_shape   = out["X_train"].shape,
        feature_names = out["X_train"].columns.tolist(),
    )

    print("\n[train] ✅ All done!")


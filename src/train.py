"""
train.py
Model training, hyperparameter search, and model selection for both Food Demand
and Food Waste prediction pipelines using time-series validation.
"""

import os
import sys
import json
import time
import joblib
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import (
    build_demand_preprocessor,
    build_waste_preprocessor,
    prepare_demand_features,
    prepare_waste_features,
    chronological_split,
    clean_dataframe
)
from src.evaluate import evaluate_model

RANDOM_SEED = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def build_pipeline_wrapper(preprocessor, model) -> Pipeline:
    """Encapsulate ColumnTransformer preprocessor and estimator into an execution Pipeline."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def get_param_grid(name: str) -> Dict[str, list]:
    """Retrieve hyperparameter search distributions for candidate estimators."""
    if name == "Decision Tree":
        return {
            "model__max_depth": [4, 6, 8, 10, 12],
            "model__min_samples_leaf": [2, 4, 8, 16],
            "model__min_samples_split": [2, 5, 10],
        }
    if name == "Random Forest":
        return {
            "model__n_estimators": [100, 150, 200],
            "model__max_depth": [6, 8, 10, 12],
            "model__min_samples_leaf": [2, 4, 8],
            "model__max_features": ["sqrt", 0.5],
        }
    if name == "Gradient Boosting":
        return {
            "model__n_estimators": [80, 120],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 4, 5],
            "model__subsample": [0.8, 1.0],
        }
    return {}


def train_model_suite(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor_builder,
    n_iter: int = 10
) -> Dict[str, Any]:
    """Train, optimize, and benchmark full suite of candidate algorithms across chronological splits."""
    candidate_estimators = {
        "Dummy Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_SEED),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_SEED),
    }

    results = {}

    for name, estimator in candidate_estimators.items():
        t0 = time.time()
        pipeline = build_pipeline_wrapper(preprocessor_builder(), estimator)
        grid = get_param_grid(name)

        if grid:
            cv = TimeSeriesSplit(n_splits=3)
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=grid,
                n_iter=n_iter,
                cv=cv,
                scoring="neg_mean_squared_error",
                refit=True,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            best_params = search.best_params_
        else:
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_params = {}

        elapsed = time.time() - t0
        val_eval = evaluate_model(best_pipeline, X_val, y_val, label=f"{name} (Val)")
        test_eval = evaluate_model(best_pipeline, X_test, y_test, label=f"{name} (Test)")

        results[name] = {
            "pipeline": best_pipeline,
            "best_params": best_params,
            "val_metrics": val_eval,
            "test_metrics": test_eval,
            "train_time_s": round(elapsed, 2),
        }

    valid_candidates = {k: v for k, v in results.items() if k != "Dummy Baseline"}
    winner_name = max(valid_candidates, key=lambda k: valid_candidates[k]["val_metrics"]["R2"])
    winner_pipeline = results[winner_name]["pipeline"]

    return {
        "results": results,
        "best_model_name": winner_name,
        "best_pipeline": winner_pipeline,
    }


def train_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_iter: int = 10
) -> Dict[str, Any]:
    """Train and optimize both demand prediction and waste prediction pipelines."""
    print("Training Food Demand Prediction Models...")
    X_d_train, y_d_train = prepare_demand_features(train_df)
    X_d_val, y_d_val = prepare_demand_features(val_df)
    X_d_test, y_d_test = prepare_demand_features(test_df)

    demand_results = train_model_suite(
        X_d_train, y_d_train, X_d_val, y_d_val, X_d_test, y_d_test,
        preprocessor_builder=build_demand_preprocessor,
        n_iter=n_iter,
    )

    print("\nTraining Food Waste Prediction Models...")
    X_w_train, y_w_train = prepare_waste_features(train_df)
    X_w_val, y_w_val = prepare_waste_features(val_df)
    X_w_test, y_w_test = prepare_waste_features(test_df)

    waste_results = train_model_suite(
        X_w_train, y_w_train, X_w_val, y_w_val, X_w_test, y_w_test,
        preprocessor_builder=build_waste_preprocessor,
        n_iter=n_iter,
    )

    return {
        "demand": demand_results,
        "waste": waste_results,
        "features": {
            "demand": X_d_train.columns.tolist(),
            "waste": X_w_train.columns.tolist(),
        },
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }


def save_models_and_metadata(training_output: Dict[str, Any]) -> Tuple[str, str, str]:
    """Serialize trained pipelines with compression and save benchmark performance metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    demand_model_path = os.path.join(MODELS_DIR, "demand_model.pkl")
    joblib.dump(training_output["demand"]["best_pipeline"], demand_model_path, compress=3)

    legacy_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(training_output["demand"]["best_pipeline"], legacy_model_path, compress=3)

    waste_model_path = os.path.join(MODELS_DIR, "waste_model.pkl")
    joblib.dump(training_output["waste"]["best_pipeline"], waste_model_path, compress=3)

    def extract_serializable(model_dict: Dict[str, Any]) -> Dict[str, Any]:
        extracted = {}
        for name, data in model_dict["results"].items():
            extracted[name] = {
                "val_metrics": {k: float(v) for k, v in data["val_metrics"].items() if k != "predictions"},
                "test_metrics": {k: float(v) for k, v in data["test_metrics"].items() if k != "predictions"},
                "train_time_s": data["train_time_s"],
                "best_params": data["best_params"],
            }
        return extracted

    metadata = {
        "demand_model": {
            "best_model_name": training_output["demand"]["best_model_name"],
            "features": training_output["features"]["demand"],
            "results": extract_serializable(training_output["demand"]),
        },
        "waste_model": {
            "best_model_name": training_output["waste"]["best_model_name"],
            "features": training_output["features"]["waste"],
            "results": extract_serializable(training_output["waste"]),
        },
        "dataset_split": {
            "train_rows": training_output["train_rows"],
            "val_rows": training_output["val_rows"],
            "test_rows": training_output["test_rows"],
        },
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return demand_model_path, waste_model_path, metadata_path


if __name__ == "__main__":
    from src.data_generation import generate_dataset

    csv_path = os.path.join(BASE_DIR, "data", "mess_dataset.csv")
    if not os.path.exists(csv_path):
        raw_df = generate_dataset(output_path=csv_path)
    else:
        raw_df = pd.read_csv(csv_path, parse_dates=["Date"])

    cleaned_df = clean_dataframe(raw_df)
    train_split, val_split, test_split = chronological_split(cleaned_df)

    output = train_all_models(train_split, val_split, test_split, n_iter=10)
    d_path, w_path, m_path = save_models_and_metadata(output)
    print(f"\nArtifacts successfully written:\n- {d_path}\n- {w_path}\n- {m_path}")

"""
evaluate.py
-----------
Model evaluation utilities for the Mess Prediction System.

Author : ML Engineering Team
Date   : 2024
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(pipeline, X, y, label: str = "") -> dict:
    """
    Evaluate a fitted Pipeline on (X, y) and return a metrics dict.

    Returns
    -------
    dict with keys: MAE, MSE, RMSE, R2, predictions
    """
    y_pred = pipeline.predict(X)
    mae  = mean_absolute_error(y, y_pred)
    mse  = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y, y_pred)

    if label:
        print(f"  [{label}] MAE={mae:.3f}  MSE={mse:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")

    return {
        "MAE":         round(mae,  4),
        "MSE":         round(mse,  4),
        "RMSE":        round(rmse, 4),
        "R2":          round(r2,   4),
        "predictions": y_pred,
    }


def get_feature_importances(pipeline, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from the best model step in the pipeline.
    Works for RandomForestRegressor and DecisionTreeRegressor.
    Returns empty DataFrame for models without feature_importances_.

    Parameters
    ----------
    pipeline      : fitted sklearn Pipeline
    feature_names : list of original feature column names

    Returns
    -------
    pd.DataFrame with columns ['Feature', 'Importance'] sorted descending
    """
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return pd.DataFrame(columns=["Feature", "Importance"])

    try:
        # Get transformed feature names
        tf_names = preprocessor.get_feature_names_out()
    except Exception:
        tf_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    importances = model.feature_importances_
    df = pd.DataFrame({
        "Feature":    tf_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df


def residual_stats(y_true, y_pred) -> dict:
    """
    Compute residual statistics for diagnostic plots.

    Returns
    -------
    dict with residuals array and summary statistics
    """
    residuals = np.array(y_true) - np.array(y_pred)
    return {
        "residuals": residuals,
        "mean":      float(np.mean(residuals)),
        "std":       float(np.std(residuals)),
        "median":    float(np.median(residuals)),
        "q5":        float(np.percentile(residuals, 5)),
        "q95":       float(np.percentile(residuals, 95)),
    }


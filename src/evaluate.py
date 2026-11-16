"""
evaluate.py
Model evaluation utilities for the University Mess Prediction System.
Computes evaluation metrics, feature importances, and residual error distributions.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(pipeline, X: pd.DataFrame, y: pd.Series, label: str = "") -> Dict[str, Any]:
    """Evaluate fitted model pipeline against ground truth targets and return metrics dictionary."""
    y_pred = pipeline.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    if label:
        print(f"  [{label}] MAE={mae:.3f}  MSE={mse:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")

    return {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "predictions": y_pred,
    }


def get_feature_importances(pipeline, feature_names: List[str] = None) -> pd.DataFrame:
    """Extract and sort feature importances from tree-based or ensemble estimators."""
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return pd.DataFrame(columns=["Feature", "Importance"])

    try:
        transformed_names = preprocessor.get_feature_names_out()
    except Exception:
        transformed_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    clean_names = [str(name).replace("num__", "").replace("ord__", "").replace("nom__", "") for name in transformed_names]

    df_imp = pd.DataFrame({
        "Feature": clean_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df_imp


def residual_stats(y_true, y_pred) -> Dict[str, Any]:
    """Compute error residual statistics including mean, standard deviation, and quantiles."""
    residuals = np.array(y_true) - np.array(y_pred)
    return {
        "residuals": residuals,
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "median": float(np.median(residuals)),
        "q5": float(np.percentile(residuals, 5)),
        "q95": float(np.percentile(residuals, 95)),
    }

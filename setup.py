"""
setup.py
--------
One-shot setup script for the University Mess Prediction System.

Steps performed:
  1. Generate synthetic dataset → data/mess_dataset.csv
  2. Initialize SQLite database
  3. Train and evaluate all ML models
  4. Save best model → models/best_model.pkl
  5. Seed database with sample records

Run from the project root:
    python setup.py

Author : ML Engineering Team
Date   : 2024
"""

import os
import sys

# Ensure src/ is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import pandas as pd

from src.data_generation import generate_dataset
from src.preprocessing   import clean_dataframe, chronological_split
from src.train           import train_all_models, save_model
from src.database        import initialize_database, bulk_insert_from_df

CSV_PATH = os.path.join(BASE_DIR, "data", "mess_dataset.csv")


def main():
    print("=" * 60)
    print("  University Mess Prediction System — Setup")
    print("=" * 60)

    # ── Step 1: Dataset ───────────────────────────────────────
    print("\n[Step 1/4] Generating synthetic dataset …")
    if os.path.exists(CSV_PATH):
        print(f"  Dataset already exists at {CSV_PATH}, loading …")
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    else:
        df = generate_dataset(output_path=CSV_PATH)

    print(f"  Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Step 2: Database ──────────────────────────────────────
    print("\n[Step 2/4] Initializing SQLite database …")
    initialize_database()
    print("  Seeding sample records …")
    try:
        bulk_insert_from_df(df, limit=2000)
    except Exception as e:
        print(f"  Warning: bulk insert skipped — {e}")

    # ── Step 3: Preprocessing + Training ─────────────────────
    print("\n[Step 3/4] Preprocessing and training models …")
    df_clean = clean_dataframe(df)
    train_df, val_df, test_df = chronological_split(df_clean)

    out = train_all_models(train_df, val_df, test_df, n_iter=15)

    # ── Step 4: Save model ────────────────────────────────────
    print("\n[Step 4/4] Saving best model and metadata …")
    model_path, meta_path = save_model(
        pipeline      = out["best_pipeline"],
        model_name    = out["best_model_name"],
        results       = out["results"],
        train_shape   = out["X_train"].shape,
        feature_names = out["X_train"].columns.tolist(),
    )

    print("\n" + "=" * 60)
    print("  ✅  Setup complete!")
    print("=" * 60)
    print(f"\n  Dataset  : {CSV_PATH}")
    print(f"  Model    : {model_path}")
    print(f"  Metadata : {meta_path}")
    print(f"\n  Best model  : {out['best_model_name']}")
    bm = out["results"][out["best_model_name"]]
    print(f"  Test MAE    : {bm['test_metrics']['MAE']}")
    print(f"  Test RMSE   : {bm['test_metrics']['RMSE']}")
    print(f"  Test R²     : {bm['test_metrics']['R2']}")
    print("\n  To start the app:")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    main()


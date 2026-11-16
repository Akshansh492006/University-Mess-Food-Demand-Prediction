"""
setup.py
Automated end-to-end setup script for the University Mess Prediction System.
Handles dataset generation, database schema initialization, seed ingestion,
and supervised model training for both demand and waste forecasting pipelines.
"""

import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_generation import generate_dataset
from src.preprocessing import clean_dataframe, chronological_split
from src.train import train_all_models, save_models_and_metadata
from src.database import initialize_database, bulk_insert_from_df

CSV_PATH = os.path.join(BASE_DIR, "data", "mess_dataset.csv")


def main():
    print("=" * 60)
    print("  University Mess Prediction System - Setup")
    print("=" * 60)

    print("\n[Step 1/4] Preparing dataset...")
    df = generate_dataset(output_path=CSV_PATH)
    print(f"  Dataset ready: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n[Step 2/4] Initializing SQLite database...")
    initialize_database()
    bulk_insert_from_df(df, limit=2000)

    print("\n[Step 3/4] Preprocessing and training models...")
    df_clean = clean_dataframe(df)
    train_df, val_df, test_df = chronological_split(df_clean)

    training_output = train_all_models(train_df, val_df, test_df, n_iter=10)

    print("\n[Step 4/4] Saving compressed models and evaluation metadata...")
    d_path, w_path, m_path = save_models_and_metadata(training_output)

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print(f"  Demand Model  : {d_path}")
    print(f"  Waste Model   : {w_path}")
    print(f"  Metadata File : {m_path}")

    d_winner = training_output["demand"]["best_model_name"]
    d_eval = training_output["demand"]["results"][d_winner]["test_metrics"]
    print(f"\n  Demand Champion : {d_winner} (Test R2={d_eval['R2']}, Test MAE={d_eval['MAE']})")

    w_winner = training_output["waste"]["best_model_name"]
    w_eval = training_output["waste"]["results"][w_winner]["test_metrics"]
    print(f"  Waste Champion  : {w_winner} (Test R2={w_eval['R2']}, Test MAE={w_eval['MAE']})")


if __name__ == "__main__":
    main()

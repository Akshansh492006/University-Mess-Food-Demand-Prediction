"""
database.py
-----------
SQLite database layer for the Mess Prediction System.

Tables:
  - food_items
  - historical_consumption
  - attendance_records
  - menu_records
  - prediction_history
  - waste_records

Author : ML Engineering Team
Date   : 2024
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "mess_system.db"
)


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: str = DB_PATH) -> None:
    """Create all tables if they do not already exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = get_connection(db_path)
    cur  = conn.cursor()

    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS food_items (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        food_name       TEXT    NOT NULL UNIQUE,
        category        TEXT    NOT NULL,
        meal_period     TEXT    NOT NULL,
        base_demand     REAL    DEFAULT 0,
        active          INTEGER DEFAULT 1,
        created_at      TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS historical_consumption (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        date             TEXT    NOT NULL,
        meal_period      TEXT    NOT NULL,
        food_item        TEXT    NOT NULL,
        food_category    TEXT,
        student_attend   INTEGER,
        exam_day         INTEGER DEFAULT 0,
        holiday          INTEGER DEFAULT 0,
        weekend          INTEGER DEFAULT 0,
        weather          TEXT,
        qty_prepared     REAL    DEFAULT 0,
        qty_consumed     REAL    DEFAULT 0,
        leftover         REAL    DEFAULT 0,
        waste_pct        REAL    DEFAULT 0,
        actual_demand    REAL    DEFAULT 0,
        created_at       TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS attendance_records (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        date            TEXT    NOT NULL,
        meal_period     TEXT    NOT NULL,
        total_students  INTEGER NOT NULL,
        present_count   INTEGER NOT NULL,
        absent_count    INTEGER,
        notes           TEXT,
        created_at      TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS menu_records (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        date            TEXT    NOT NULL,
        meal_period     TEXT    NOT NULL,
        food_item       TEXT    NOT NULL,
        planned_qty     REAL,
        actual_qty      REAL,
        notes           TEXT,
        created_at      TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS prediction_history (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        date                TEXT,
        meal_period         TEXT,
        food_item           TEXT,
        food_category       TEXT,
        student_attendance  INTEGER,
        exam_day            INTEGER,
        holiday             INTEGER,
        weather             TEXT,
        predicted_demand    REAL,
        suggested_qty       REAL,
        estimated_surplus   REAL,
        model_name          TEXT,
        created_at          TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS waste_records (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        date            TEXT,
        meal_period     TEXT,
        food_item       TEXT,
        qty_prepared    REAL,
        qty_consumed    REAL,
        waste_qty       REAL,
        waste_pct       REAL,
        notes           TEXT,
        created_at      TEXT    DEFAULT (datetime('now'))
    );
    """)

    conn.commit()
    conn.close()
    print(f"[database] Initialized → {db_path}")


# ── CRUD helpers ─────────────────────────────────────────────

def insert_prediction(record: dict, db_path: str = DB_PATH) -> int:
    """Insert a prediction record. Returns the new row id."""
    conn = get_connection(db_path)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO prediction_history
        (date, meal_period, food_item, food_category, student_attendance,
         exam_day, holiday, weather, predicted_demand, suggested_qty,
         estimated_surplus, model_name, created_at)
        VALUES (:date, :meal_period, :food_item, :food_category,
                :student_attendance, :exam_day, :holiday, :weather,
                :predicted_demand, :suggested_qty, :estimated_surplus,
                :model_name, :created_at)
    """, {**record, "created_at": datetime.now().isoformat()})
    rowid = cur.lastrowid
    conn.commit()
    conn.close()
    return rowid


def fetch_predictions(limit: int = 100, db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch recent predictions as DataFrame."""
    conn = get_connection(db_path)
    df   = pd.read_sql_query(
        f"SELECT * FROM prediction_history ORDER BY id DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df


def insert_consumption(record: dict, db_path: str = DB_PATH) -> int:
    """Insert a historical consumption record."""
    conn = get_connection(db_path)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO historical_consumption
        (date, meal_period, food_item, food_category, student_attend,
         exam_day, holiday, weekend, weather, qty_prepared,
         qty_consumed, leftover, waste_pct, actual_demand)
        VALUES (:date, :meal_period, :food_item, :food_category,
                :student_attend, :exam_day, :holiday, :weekend,
                :weather, :qty_prepared, :qty_consumed, :leftover,
                :waste_pct, :actual_demand)
    """, record)
    rowid = cur.lastrowid
    conn.commit()
    conn.close()
    return rowid


def fetch_consumption(limit: int = 500, db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch historical consumption records."""
    conn = get_connection(db_path)
    df   = pd.read_sql_query(
        f"SELECT * FROM historical_consumption ORDER BY date DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df


def insert_waste_record(record: dict, db_path: str = DB_PATH) -> int:
    """Insert a waste record."""
    conn = get_connection(db_path)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO waste_records
        (date, meal_period, food_item, qty_prepared, qty_consumed,
         waste_qty, waste_pct, notes)
        VALUES (:date, :meal_period, :food_item, :qty_prepared,
                :qty_consumed, :waste_qty, :waste_pct, :notes)
    """, record)
    rowid = cur.lastrowid
    conn.commit()
    conn.close()
    return rowid


def fetch_waste_records(limit: int = 500, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = get_connection(db_path)
    df   = pd.read_sql_query(
        f"SELECT * FROM waste_records ORDER BY date DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df


def get_food_items(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = get_connection(db_path)
    df   = pd.read_sql_query("SELECT * FROM food_items WHERE active=1", conn)
    conn.close()
    return df


def bulk_insert_from_df(df: pd.DataFrame, db_path: str = DB_PATH,
                        limit: int = 2000) -> None:
    """
    Bulk-insert the first `limit` rows from the dataset CSV into
    historical_consumption for demonstration.
    """
    conn = get_connection(db_path)
    sample = df.head(limit).copy()
    sample = sample.rename(columns={
        "Student_Attendance":   "student_attend",
        "Examination_Day":      "exam_day",
        "Weekend":              "weekend",
        "Holiday":              "holiday",
        "Quantity_Prepared":    "qty_prepared",
        "Quantity_Consumed":    "qty_consumed",
        "Leftover_Quantity":    "leftover",
        "Food_Waste_Percentage":"waste_pct",
        "Actual_Food_Demand":   "actual_demand",
        "Meal_Period":          "meal_period",
        "Food_Item":            "food_item",
        "Food_Category":        "food_category",
        "Weather":              "weather",
        "Date":                 "date",
    })
    cols = ["date","meal_period","food_item","food_category","student_attend",
            "exam_day","holiday","weekend","weather","qty_prepared",
            "qty_consumed","leftover","waste_pct","actual_demand"]
    sample = sample[[c for c in cols if c in sample.columns]]
    sample["date"] = sample["date"].astype(str)
    sample.to_sql("historical_consumption", conn,
                  if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"[database] Inserted {len(sample)} rows into historical_consumption")


if __name__ == "__main__":
    initialize_database()


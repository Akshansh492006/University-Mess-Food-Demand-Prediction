# 🍽️ University Mess Food Demand & Waste Prediction System

> **INT395 Academic Project** | Machine Learning · Python · Streamlit  
> ⚠️ **All data is entirely synthetic** — generated algorithmically for demonstration purposes only. It does not represent actual university mess records.

---

## 📋 Project Introduction

The University Mess Food Demand & Waste Prediction System is a supervised machine learning application that estimates the quantity of food required for different meal periods in a university mess and helps minimise food wastage. It provides mess administrators with data-driven demand forecasts, waste analytics, and an interactive dashboard — replacing manual guesswork with model-backed predictions.

---

## ❗ Problem Statement

University mess kitchens routinely prepare food quantities based on experience and intuition, leading to two persistent problems:

| Problem | Consequence |
|---------|-------------|
| **Over-preparation** | Large quantities of food wasted daily |
| **Under-preparation** | Students go without food; dissatisfaction |

Neither problem has a simple fixed-rule solution because demand varies daily based on attendance, examinations, holidays, weather, food type, and seasonal patterns.

---

## 🎯 Objectives

1. Build a **supervised ML model** that accurately predicts per-item food demand for each meal period.
2. Identify the **key factors** influencing mess demand (feature importance).
3. Provide an **interactive dashboard** that mess managers can use daily.
4. Quantify **food waste** and suggest actionable reduction strategies.
5. Maintain a **historical database** of consumption and predictions.

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.12 |
| Data | Pandas, NumPy |
| ML | Scikit-learn |
| Algorithms | Linear Regression, Decision Tree, Random Forest, Dummy |
| Visualisation | Plotly, Matplotlib |
| Web App | Streamlit |
| Storage | SQLite + Joblib |
| Testing | pytest |

---

## 📊 Dataset Description

> **This is a synthetic dataset.** It was algorithmically generated using realistic mathematical relationships.

| Field | Description |
|-------|-------------|
| `Date` | Date of the record (2023-01-01 to 2024-12-31) |
| `Day_of_Week` | Monday … Sunday |
| `Month` | 1 – 12 |
| `Meal_Period` | Breakfast / Lunch / Dinner |
| `Food_Item` | Specific dish (28 items across 3 meal periods) |
| `Food_Category` | Staples, North Indian, South Indian, etc. |
| `Student_Attendance` | Number of students present |
| `Examination_Day` | Binary flag |
| `Holiday` | Binary flag |
| `Weekend` | Binary flag |
| `Weather` | Sunny / Cloudy / Rainy / Cold / Hot |
| `Previous_Day_Demand` | Demand from the previous day |
| `Previous_Week_Demand` | Demand from the same day last week |
| `Historical_Avg_Demand` | 30-day rolling average demand |
| `Actual_Food_Demand` | **Target variable** — actual demand in units |
| `Quantity_Prepared` | Quantity prepared by kitchen (with buffer) |
| `Quantity_Consumed` | Quantity actually consumed |
| `Leftover_Quantity` | Quantity_Prepared − Quantity_Consumed |
| `Food_Waste_Percentage` | (Leftover / Prepared) × 100 |

**Size:** ≈ 10,920+ records (3 meals × 28 items × 730 days)

**Key simulated relationships:**
- Higher attendance → higher demand
- Exam days / holidays → significantly reduced demand
- Rainy weather → lower turnout
- Seasonal cosine pattern peaks in October–November (mid-semester)
- Lag features (previous day, previous week, 30-day average) capture temporal patterns

---

## 🤖 Machine Learning Methodology

### Split Strategy
Chronological (no shuffling) to prevent temporal data leakage:
- **Training set:** First 70% of records
- **Validation set:** Next 15% (model selection)
- **Test set:** Final 15% (unbiased evaluation)

### Models Trained
| Model | Purpose |
|-------|---------|
| Dummy Regressor (mean) | Baseline |
| Linear Regression | Simple linear benchmark |
| Decision Tree Regressor | Non-linear, interpretable |
| Random Forest Regressor | Ensemble — typically best |

### Hyperparameter Optimisation
`RandomizedSearchCV` with `TimeSeriesSplit(n_splits=5)` applied on the training set for tree-based models.

### Preprocessing Pipeline
- **Numeric features** → `SimpleImputer(median)` + `StandardScaler`
- **Day-of-Week** → `OrdinalEncoder` (Monday=0 … Sunday=6)
- **Nominal features** → `OneHotEncoder(handle_unknown='ignore')`
- Assembled with `ColumnTransformer` inside a `sklearn.Pipeline`
- **Target-leaking columns** (`Quantity_Prepared`, `Quantity_Consumed`, `Leftover_Quantity`, `Food_Waste_Percentage`) are excluded from features.

---

## 📈 Model Evaluation Metrics

Each model is evaluated on the validation and test sets using:

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **R²** | Coefficient of Determination |

Actual evaluation results (not hardcoded) are stored in `models/model_metadata.json` and displayed on the ML Insights page.

---

## 🗑️ Food Waste Analytics

```
Food Waste    = Quantity_Prepared - Quantity_Consumed
Waste %       = (Food Waste / Quantity_Prepared) × 100
```

The waste module reports:
- Total waste across all records
- Most wasted food items (by total and by average %)
- Meal-wise waste breakdown
- Monthly waste trends
- Estimated cost savings if waste were reduced to 5%
- Weekly pattern of waste

---

## 🗂️ Project Structure

```
mess_prediction_system/
│
├── app.py                      # Streamlit entry point
├── setup.py                    # One-shot setup script
├── requirements.txt
├── README.md
│
├── data/
│   ├── mess_dataset.csv        # Synthetic dataset (generated)
│   └── mess_system.db          # SQLite database
│
├── models/
│   ├── best_model.pkl          # Saved best model (joblib)
│   └── model_metadata.json     # Evaluation results & config
│
├── src/
│   ├── data_generation.py      # Synthetic data generator
│   ├── preprocessing.py        # Cleaning, pipelines, splits
│   ├── train.py                # Training + hyperparameter search
│   ├── evaluate.py             # Metrics and diagnostics
│   ├── predict.py              # Prediction engine
│   ├── waste_analysis.py       # Waste analytics helpers
│   └── database.py             # SQLite CRUD layer
│
├── pages/                      # Streamlit multi-page app
│   ├── 1_Dashboard.py
│   ├── 2_Demand_Prediction.py
│   ├── 3_Waste_Analytics.py
│   ├── 4_ML_Insights.py
│   ├── 5_Data_Management.py
│   └── 6_Reports.py
│
├── tests/
│   └── test_system.py          # pytest unit + integration tests
│
├── reports/                    # Downloaded report CSVs saved here
└── assets/                     # Static assets
```

---

## ⚙️ Installation Instructions

### Prerequisites
- Python 3.10 or later
- pip

### Steps

```bash
# 1. Navigate to the project folder
cd "mess_prediction_system"

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the one-shot setup (generates data, trains models, seeds DB)
python setup.py

# 5. Launch the Streamlit app
streamlit run app.py
```

---

## 🚀 How to Run the Application

After running `setup.py` once:

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

### Pages

| Page | URL Fragment | Purpose |
|------|-------------|---------|
| Home | `/` | Landing page |
| Overview Dashboard | `1_Dashboard` | KPIs, trends, meal stats |
| Food Demand Prediction | `2_Demand_Prediction` | Live ML predictions |
| Food Waste Analytics | `3_Waste_Analytics` | Waste breakdown & trends |
| ML Insights | `4_ML_Insights` | Model comparison, feature importance |
| Data Management | `5_Data_Management` | View, add, download records |
| Reports | `6_Reports` | Daily / weekly / monthly reports |

### Running Tests

```bash
python -m pytest tests/ -v
```

---

## 🔭 Future Scope

| Enhancement | Description |
|-------------|-------------|
| Real data integration | Connect to actual university ERP / attendance systems |
| Deep learning | LSTM / Transformer models for temporal forecasting |
| Menu optimisation | Recommend optimal menus to minimise waste |
| Mobile app | React Native / Flutter companion app |
| Alert system | Email / SMS alerts when predicted demand spikes |
| Multi-mess support | Extend to multiple mess halls across a campus |
| Cost modelling | Integrate ingredient prices for financial forecasting |
| Nutritional analysis | Track nutritional content of prepared meals |

---

## 📌 Notes for Evaluators

- All model performance figures are **computed at runtime** — never hardcoded.
- The dataset is **labelled as synthetic** throughout the application UI.
- The chronological split prevents temporal data leakage.
- Target-leaking columns are explicitly excluded from model features.
- The application integrates the **saved trained model** — not rule-based fallbacks.

---

*INT395 Academic Project — University Mess Food Demand & Waste Prediction System*


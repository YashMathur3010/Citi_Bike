# Citi Bike Trip Prediction System

This project aims to build an end-to-end machine learning pipeline for predicting Citi Bike trip demand using public trip history data. The system is adapted from a previous NYC taxi demand prediction project and tailored specifically to the structure, granularity, and challenges of Citi Bike data.

## 📅 Project Objective

To forecast hourly Citi Bike trip volumes at the station or zone level using historical usage patterns, with potential applications in:

* Bike redistribution and operations planning
* Infrastructure optimization
* Real-time dashboarding and alerts

## 🚋 Data Source

We use publicly available trip history data provided by Citi Bike:

* **Data Index**: [https://s3.amazonaws.com/tripdata/index.html](https://s3.amazonaws.com/tripdata/index.html)

### Data Features (sample fields):

* `starttime`, `stoptime`
* `start station id`, `end station id`
* `start station name`, `end station name`
* `bikeid`, `usertype`, `birth year`, `gender`

Each file contains data for one month and is available in CSV or ZIP format. Columns may vary slightly over time.

## 🎯 Target

- Predict the number of **bike trips per hour** (time-series regression)
- Granularity: global or optionally by station/neighborhood

---

## 🧱 Project Structure (Base Template)

```
citibike_trip_prediction/
├── data/                  # Raw and processed data
├── notebooks/             # Jupyter notebooks per ML phase
├── pipelines/             # Training and inference pipelines
├── src/                   # Core Python modules
├── test/                  # Unit tests
├── frontend/              # (Optional) frontend dashboard
├── requirements.txt       # Python dependencies
├── README.md              # Project overview (this file)
└── vscode_config.json     # Optional editor settings
```

---

## 🚀 Project Phases

### Phase 0: Initialization
- Set up repo and project structure
- Create base configuration (`src/config.py`)
- Write this README

### Phase 1: Data Ingestion
- Download raw monthly data from S3
- Validate structure and store clean files

### Phase 2: Time-Series Aggregation
- Resample trips to hourly buckets
- Generate the target time series

### Phase 3: Feature Engineering
- Create lag features, rolling means, time encodings, etc.

### Phase 4: Model Training
- Train models (e.g., LightGBM) on historical data
- Evaluate using time-aware validation

### Phase 5: Inference
- Use trained model to make hourly forecasts
- Save predictions for dashboard or alerts

### Phase 6: Frontend Dashboard
- Visualize actuals vs. forecasts using Streamlit/Flask

### Phase 7: Automation
- Add scripts for retraining and batch predictions
- Schedule via cronjob/Airflow

---

## ⚙️ Technologies

- Python 3.11
- Pandas, NumPy
- XGBoost, LightGBM
- Jupyter
- Streamlit
- GitHub Actions

---

## 📅 Timeline

The project will be implemented iteratively across the 8 phases outlined above. Each phase builds on the previous to enable end-to-end learning and production readiness.

---

## 👤 Author

Adapted by Yash Mathur
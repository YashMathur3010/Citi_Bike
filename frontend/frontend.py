import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# Set up path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Standard imports
from src.inference import (
    fetch_next_hour_predictions,
    load_batch_of_features_from_store,
    fetch_hourly_rides,
    fetch_predictions,
    load_model_from_registry,
    get_model_predictions,
)
from src.plot_utils import plot_prediction, plot_aggregated_time_series

# Streamlit UI setup
st.set_page_config(page_title="Citi Bike Predictions", layout="wide")
st.title("🚴 Citi Bike Ride Demand Monitor & Predictor")

# ─────────────────────────────
# 📊 Section 1: Latest Forecasts
# ─────────────────────────────
st.header("📊 Latest 1-Hour Forecasts")
latest_preds = fetch_next_hour_predictions()
if not latest_preds.empty:
    st.dataframe(latest_preds, use_container_width=True)
else:
    st.warning("No latest predictions available.")

# ─────────────────────────────────────────────
# 🔍 Section 2: Historical Pattern + Prediction
# ─────────────────────────────────────────────
st.header("🔍 Historical Pattern + Prediction")

current_time = pd.Timestamp.now(tz="Etc/UTC")
features = load_batch_of_features_from_store(current_time)

if features.empty:
    st.warning("Not enough time-series data to generate features.")
else:
    row_id = st.slider("Select Forecast Row ID", min_value=0, max_value=len(features) - 1, value=0)

    if st.button("Show Forecast Plot"):
        model = load_model_from_registry()
        prediction = get_model_predictions(model, features.iloc[[row_id]])
        fig = plot_prediction(features.iloc[[row_id]], prediction)
        st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────
# 📉 Section 3: Past MAE Monitoring
# ────────────────────────────────
st.header("📉 Mean Absolute Error Over Time")

past_hours = st.slider("Select History Window (Hours)", 6, 240, 24, step=6)
rides = fetch_hourly_rides(past_hours)
preds = fetch_predictions(past_hours)

if not rides.empty and not preds.empty:
    merged = pd.merge(rides, preds, on=["start_station_id", "hour"])
    merged["absolute_error"] = abs(merged["ride_count"] - merged["predicted_demand"])

    error_fig = px.line(
        merged,
        x="hour",
        y="absolute_error",
        color="start_station_id",
        title="Prediction Error by Station",
        labels={"hour": "Time", "absolute_error": "MAE"}
    )
    st.plotly_chart(error_fig, use_container_width=True)
else:
    st.info("Not enough data to show error trends.")

# Footer
st.markdown("---")
st.caption("© 2025 Citi Bike Forecasting Dashboard | Streamlit + Hopsworks")

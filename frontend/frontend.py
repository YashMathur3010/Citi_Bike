import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import (
    fetch_next_hour_predictions,
    load_batch_of_features_from_store,
    fetch_hourly_rides,
    fetch_predictions
)
from src.plot_utils import plot_prediction, plot_aggregated_time_series

st.set_page_config(page_title="Citi Bike Predictions", layout="wide")
st.title("🚴 Citi Bike Ride Demand Monitor & Predictor")

# --- Section 1: Latest Predictions
st.header("📊 Latest Forecasted Demand")

latest_preds = fetch_next_hour_predictions()
st.dataframe(latest_preds, use_container_width=True)

# --- Section 2: Individual Forecast Visualization
st.header("🔍 Visualize Historical Pattern + Forecast")

features = load_batch_of_features_from_store(pd.Timestamp.now(tz="Etc/UTC"))

row_id = st.slider("Select Forecast Row ID", min_value=0, max_value=len(features)-1, value=0)
model = None  # placeholder if needed

# If user clicks, show prediction plot
if st.button("Show Forecast Plot"):
    from src.inference import load_model_from_registry, get_model_predictions
    model = load_model_from_registry()
    prediction = get_model_predictions(model, features.iloc[[row_id]])
    fig = plot_prediction(features.iloc[[row_id]], prediction)
    st.plotly_chart(fig, use_container_width=True)

# --- Section 3: Past MAE Monitoring
st.header("📉 Mean Absolute Error Over Time")

past_hours = st.slider("Select Number of Hours", 6, 240, 24, step=6)
rides = fetch_hourly_rides(past_hours)
preds = fetch_predictions(past_hours)

merged = pd.merge(rides, preds, on=["start_station_id", "hour"])
merged["absolute_error"] = abs(merged["ride_count"] - merged["predicted_demand"])

error_fig = px.line(
    merged,
    x="hour",
    y="absolute_error",
    color="start_station_id",
    title="Prediction Error by Station (last hours)"
)
st.plotly_chart(error_fig, use_container_width=True)

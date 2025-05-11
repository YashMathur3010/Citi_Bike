import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# Add project path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Keep existing function names from your original app
from src.inference import (
    fetch_predictions,
    fetch_hourly_rides,
    load_batch_of_features_from_store,
    fetch_next_hour_predictions
)
from src.plot_utils import plot_aggregated_time_series

# Page settings
st.set_page_config(page_title="Citi Bike Prediction Dashboard", layout="wide")
st.title("🚲 Citi Bike Demand Forecasting")

st.markdown("This dashboard shows predictions for hourly ride demand at top Citi Bike stations.")

# Load predictions
predictions = fetch_predictions()

if predictions.empty:
    st.warning("No predictions available yet. Try again after the pipeline runs.")
    st.stop()

# Dropdown to choose station
station_ids = predictions["start_station_id"].unique()
selected_station = st.selectbox("Select Start Station ID", sorted(station_ids))

# Filter predictions for the selected station
station_df = predictions[predictions["start_station_id"] == selected_station].sort_values("prediction_time")

# Display top 10 predictions
st.subheader(f"📊 Top 10 Predicted Rides – Station: `{selected_station}`")
st.dataframe(
    station_df[["prediction_time", "predicted_demand"]].head(10).rename(columns={
        "prediction_time": "Prediction Time",
        "predicted_demand": "Predicted Ride Count"
    }),
    use_container_width=True
)

# Plot predictions
fig = px.line(
    station_df.head(10),
    x="prediction_time",
    y="predicted_demand",
    title=f"Predicted Demand Trend – Station {selected_station}",
    markers=True,
    labels={"prediction_time": "Time", "predicted_demand": "Predicted Rides"}
)
st.plotly_chart(fig, use_container_width=True)

# Optional: plot actuals vs predictions for a specific sample
st.subheader("🕒 Historical Ride Series with Actuals and Predictions")
try:
    # Use default row_id or allow manual override
    row_id = st.number_input("Select Prediction Row Index", min_value=0, max_value=len(predictions) - 1, value=0)
    hourly_rides = fetch_hourly_rides()
    fig2 = plot_aggregated_time_series(
        hourly_rides,
        predictions["predicted_demand"],
        row_id=row_id,
        predictions=predictions["predicted_demand"]
    )
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error(f"Could not generate plot: {e}")

st.markdown("---")
st.caption("© 2025 Citi Bike ML Project · Streamlit UI powered by Hopsworks + GitHub Actions")

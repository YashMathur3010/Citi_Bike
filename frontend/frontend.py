import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# Add project path for imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from src.plot_utils import plot_aggregated_time_series
#from src.inference import get_latest_predictions
from src.inference import fetch_predictions as get_latest_predictions

st.set_page_config(page_title="Citi Bike Prediction Dashboard", layout="wide")
st.title("🚲 Citi Bike Demand Forecasting")

st.markdown("This dashboard shows **hourly ride demand predictions** using LightGBM trained on historical Citi Bike data.")

# Load predictions
predictions = get_latest_predictions()

if predictions.empty:
    st.warning("No predictions available. Please check your inference pipeline.")
    st.stop()

# Let user select station from dropdown
unique_stations = predictions["start_station_id"].unique()
selected_station = st.selectbox("Select Station ID", sorted(unique_stations))

# Filter data for selected station
station_preds = predictions[predictions["start_station_id"] == selected_station].copy()
station_preds.sort_values("prediction_time", inplace=True)

# Show top 10 predictions
st.subheader(f"📈 Next 10 Hourly Predictions for Station: `{selected_station}`")
st.dataframe(station_preds[["prediction_time", "predicted_demand"]].head(10).rename(columns={
    "prediction_time": "Prediction Time",
    "predicted_demand": "Predicted Rides"
}), use_container_width=True)

# Optional: show a line chart
fig = px.line(
    station_preds.head(10),
    x="prediction_time",
    y="predicted_demand",
    title=f"Predicted Demand Trend – Station {selected_station}",
    markers=True,
    labels={"prediction_time": "Time", "predicted_demand": "Predicted Rides"}
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("© 2025 Citi Bike ML Forecasting. Powered by Hopsworks + GitHub Actions.")

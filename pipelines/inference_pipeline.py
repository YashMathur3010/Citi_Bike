from datetime import datetime, timedelta
import pandas as pd

import src.config as config
from src.inference import (
    get_feature_store,
    get_model_predictions,
    load_model_from_registry,
)

# Get the current UTC datetime
current_date = pd.Timestamp.now(tz="Etc/UTC")
feature_store = get_feature_store()

# Define fetch range
fetch_data_to = current_date - timedelta(hours=1)
fetch_data_from = current_date - timedelta(days=29)
print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# Load feature view
feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME,
    version=config.FEATURE_VIEW_VERSION
)

# Retrieve ts_data from Hopsworks
ts_data = feature_view.get_batch_data(
    start_time=(fetch_data_from - timedelta(days=1)),
    end_time=(fetch_data_to + timedelta(days=1)),
)
ts_data = ts_data[ts_data["hour"].between(fetch_data_from, fetch_data_to)]
ts_data = ts_data.sort_values(["start_station_id", "hour"]).reset_index(drop=True)
ts_data["hour"] = ts_data["hour"].dt.tz_localize(None)

# Transform to sliding window features
from src.data_utils import transform_ts_data_info_features
features = transform_ts_data_info_features(
    ts_data, window_size=24 * 28, step_size=23
)

# Load model from registry and generate predictions
model = load_model_from_registry()
predictions = get_model_predictions(model, features)

# Show results
print(predictions.head())

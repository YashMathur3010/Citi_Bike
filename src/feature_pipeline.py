import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import hopsworks
import pandas as pd

import src.config as config
from src.data_utils import fetch_batch_raw_data, transform_raw_data_into_ts_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Step 1: Get the current datetime in UTC
current_date = pd.to_datetime(datetime.now(timezone.utc)).ceil("h")
logger.info(f"Current date and time (UTC): {current_date}")

# Step 2: Define time range for batch fetch (last 28 days)
fetch_data_to = current_date
fetch_data_from = current_date - timedelta(days=28)
logger.info(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# Step 3: Fetch raw Citi Bike data (shifted 52 weeks back)
logger.info("Fetching raw Citi Bike data...")
rides = fetch_batch_raw_data(fetch_data_from, fetch_data_to)
logger.info(f"Raw data fetched. Number of records: {len(rides)}")

# Step 4: Convert raw ride logs to hourly time-series format
logger.info("Transforming raw data into time-series format...")
ts_data = transform_raw_data_into_ts_data(rides)
logger.info(f"Transformation complete. Number of records in ts_data: {len(ts_data)}")

# Step 5: Connect to Hopsworks
logger.info("Connecting to Hopsworks project...")
project = hopsworks.login(
    project=config.HOPSWORKS_PROJECT_NAME,
    api_key_value=config.HOPSWORKS_API_KEY
)
logger.info("Connected to Hopsworks.")

# Step 6: Access the feature store
logger.info("Accessing the feature store...")
feature_store = project.get_feature_store()
logger.info("Connected to feature store.")

# Step 7: Connect to the existing feature group
logger.info(
    f"Connecting to feature group: {config.FEATURE_GROUP_NAME} (version {config.FEATURE_GROUP_VERSION})..."
)
feature_group = feature_store.get_feature_group(
    name=config.FEATURE_GROUP_NAME,
    version=config.FEATURE_GROUP_VERSION
)
logger.info("Feature group connected.")

# Step 8: Insert data into the feature group
logger.info("Inserting time-series data into feature group...")
feature_group.insert(ts_data, write_options={"wait_for_job": False})
logger.info("Data insertion completed successfully.")

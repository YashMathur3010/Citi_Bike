from datetime import datetime, timedelta, timezone

import hopsworks
import numpy as np
import pandas as pd
from hsfs.feature_store import FeatureStore

import src.config as config
from src.data_utils import transform_ts_data_into_features


def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
    )


def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(features)

    results = pd.DataFrame()
    results["start_station_id"] = features["start_station_id"].values
    results["predicted_demand"] = predictions.round(0)

    return results


def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    feature_store = get_feature_store()

    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
    )

    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
    )
    ts_data = ts_data[ts_data["hour"].between(fetch_data_from, fetch_data_to)]

    ts_data.sort_values(by=["start_station_id", "hour"], inplace=True)

    features = transform_ts_data_into_features(
        ts_data, window_size=24 * 28, step_size=23
    )

    return features


def load_model_from_registry(version=None):
    from pathlib import Path
    import joblib

    from src.pipeline_utils import TemporalFeatureEngineer, average_rides_last_4_weeks

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    models = model_registry.get_models(name=config.MODEL_NAME)
    model = max(models, key=lambda model: model.version)
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / "lgb_model.pkl")

    return model


def load_metrics_from_registry(version=None):
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()
    models = model_registry.get_models(name=config.MODEL_NAME)
    model = max(models, key=lambda model: model.version)
    return model.training_metrics


def fetch_next_hour_predictions():
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
    df = fg.read()
    df = df[df["hour"] == next_hour]

    print(f"Current UTC time: {now}")
    print(f"Next hour: {next_hour}")
    print(f"Found {len(df)} records")
    return df


def fetch_predictions(hours):
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)

    df = fg.filter((fg.hour >= current_hour)).read()
    return df


def fetch_hourly_rides(hours):
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    query = fg.select_all()
    query = query.filter(fg.hour >= current_hour)

    return query.read()


def fetch_days_data(days):
    current_date = pd.to_datetime(datetime.now(timezone.utc))
    fetch_data_from = current_date - timedelta(days=(365 + days))
    fetch_data_to = current_date - timedelta(days=365)

    print(fetch_data_from, fetch_data_to)

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    df = fg.select_all().read()
    cond = (df["hour"] >= fetch_data_from) & (df["hour"] <= fetch_data_to)
    return df[cond]

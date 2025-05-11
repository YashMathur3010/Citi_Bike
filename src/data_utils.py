import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def fetch_raw_trip_data(year: int, month: int) -> Path:
    """
    Downloads Citi Bike data for a given year and month, trying multiple filename formats.
    Saves the merged raw data as a Parquet file in RAW_DATA_DIR.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file_path = RAW_DATA_DIR / f'citi_bike_rides_raw_{year}_{month:02}.parquet'

    file_names = [
        f"{year}{month:02}-citibike-tripdata.csv.zip",
        f"{year}{month:02}-citibike-tripdata.zip"
    ]
    url_base = "https://s3.amazonaws.com/tripdata/"

    response = None
    for fname in file_names:
        url = url_base + fname
        print(f"Trying to fetch: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            print(f"✅ Successfully accessed: {url}")
            break
        else:
            print(f"❌ Failed to access: {url}")
    else:
        raise Exception(f"No valid Citi Bike data found for {year}-{month:02}.")

    expected_columns = [
        'ride_id', 'rideable_type', 'started_at', 'ended_at',
        'start_station_name', 'start_station_id',
        'end_station_name', 'end_station_id',
        'start_lat', 'start_lng', 'end_lat', 'end_lng',
        'member_casual'
    ]

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        csv_files = [file for file in zip_file.namelist() if file.endswith('.csv')]
        if not csv_files:
            raise Exception("No CSV files found inside the ZIP archive.")

        dfs = []
        for file in csv_files:
            with zip_file.open(file) as f:
                df = pd.read_csv(f, encoding='latin1', on_bad_lines='skip')
                df = df[[col for col in expected_columns if col in df.columns]]
                dfs.append(df)

        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.dropna(inplace=True)

        merged_df['start_station_id'] = merged_df['start_station_id'].astype(str)
        merged_df['end_station_id'] = merged_df['end_station_id'].astype(str)
        merged_df['started_at'] = pd.to_datetime(merged_df['started_at'], errors='coerce')
        merged_df['ended_at'] = pd.to_datetime(merged_df['ended_at'], errors='coerce')
        merged_df['start_lat'] = merged_df['start_lat'].astype(float, errors='ignore')
        merged_df['start_lng'] = merged_df['start_lng'].astype(float, errors='ignore')
        merged_df['end_lat'] = merged_df['end_lat'].astype(float, errors='ignore')
        merged_df['end_lng'] = merged_df['end_lng'].astype(float, errors='ignore')
        merged_df['member_casual'] = merged_df['member_casual'].astype(str)

        merged_df.to_parquet(output_file_path, engine='pyarrow', index=False)
        print(f"✅ Saved raw file to: {output_file_path}")
        return output_file_path


def filter_citibike_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Filters rides to those within the specified month and reasonable trip duration (<= 5 hours),
    and saves the filtered data to PROCESSED_DATA_DIR.
    """
    rides['duration'] = rides['ended_at'] - rides['started_at']
    duration_filter = (rides['duration'] > pd.Timedelta(0)) & (rides['duration'] <= pd.Timedelta(hours=5))

    start_date = pd.Timestamp(f"{year}-{month:02}-01")
    end_date = pd.Timestamp(f"{year + 1}-01-01") if month == 12 else pd.Timestamp(f"{year}-{month + 1:02}-01")
    date_filter = (rides['started_at'] >= start_date) & (rides['started_at'] < end_date)
    final_filter = duration_filter & date_filter

    filtered = rides[final_filter].copy()
    filtered = filtered[['started_at', 'start_station_id']]

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / f"citi_bike_rides_processed_{year}_{month:02}.parquet"
    filtered.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"✅ Saved filtered data to: {output_path}")

    return filtered


def load_and_process_citibike_data(year: int, months: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Loads raw Citi Bike data for the given year/months, filters it, saves processed output,
    and returns the combined filtered DataFrame.
    """
    if months is None:
        months = list(range(1, 13))

    monthly_rides = []

    for month in months:
        file_path = RAW_DATA_DIR / f'citi_bike_rides_raw_{year}_{month:02}.parquet'
        try:
            if not file_path.exists():
                print(f"⬇️ Downloading data for {year}-{month:02}...")
                fetch_raw_trip_data(year, month)
            else:
                print(f"📁 File already exists: {file_path.name}")

            print(f"📖 Loading data from: {file_path.name}")
            rides = pd.read_parquet(file_path, engine='pyarrow')

            print(f"🔍 Filtering data for {year}-{month:02}...")
            filtered = filter_citibike_data(rides, year, month)
            monthly_rides.append(filtered)

        except Exception as e:
            print(f"⚠️ Skipping {year}-{month:02} due to error: {str(e)}")
            continue

    if not monthly_rides:
        raise Exception(f"No valid data loaded for {year} and months {months}")

    combined = pd.concat(monthly_rides, ignore_index=True)
    print("✅ Successfully combined all filtered months.")
    return combined


def fill_missing_rides_full_range(df: pd.DataFrame, hour_col: str, location_col: str, rides_col: str) -> pd.DataFrame:
    """
    Ensures every (hour, station) combination is present in the time series,
    filling missing entries with ride_count = 0.
    """
    hours = pd.date_range(df[hour_col].min(), df[hour_col].max(), freq='H')
    stations = df[location_col].unique()
    full_index = pd.MultiIndex.from_product([hours, stations], names=[hour_col, location_col])
    return (
        df.set_index([hour_col, location_col])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )


def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates filtered Citi Bike ride logs into a dense time series format
    for the top 3 most used start stations.
    """
    rides['hour'] = pd.to_datetime(rides['started_at']).dt.floor('H')
    top_stations = rides['start_station_id'].value_counts().nlargest(3).index.tolist()
    rides_top = rides[rides['start_station_id'].isin(top_stations)].copy()

    grouped = (
        rides_top.groupby(['hour', 'start_station_id'])
        .size()
        .reset_index(name='ride_count')
    )

    ts_data = fill_missing_rides_full_range(
        grouped,
        hour_col='hour',
        location_col='start_station_id',
        rides_col='ride_count'
    )

    return ts_data


def transform_ts_data_into_features_and_target_loop(
    df: pd.DataFrame,
    feature_col: str = "ride_count",
    window_size: int = 12,
    step_size: int = 1
) -> tuple:
    """
    Transforms time series data for all unique start_station_id values into a tabular dataset using
    a sliding window approach.

    Args:
        df (pd.DataFrame): Time series data with 'hour', 'start_station_id', and feature_col.
        feature_col (str): Column name containing the value to use (default: 'ride_count').
        window_size (int): Number of past observations to use as features.
        step_size (int): Number of rows to slide the window by.

    Returns:
        tuple: (features_df, targets_series)
    """
    station_ids = df["start_station_id"].unique()
    transformed_data = []

    for station_id in station_ids:
        try:
            station_data = df[df["start_station_id"] == station_id].reset_index(drop=True)
            values = station_data[feature_col].values
            times = station_data["hour"].values

            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            rows = []
            for i in range(0, len(values) - window_size, step_size):
                features = values[i : i + window_size]
                target = values[i + window_size]
                target_time = times[i + window_size]

                row = np.append(features, [target, station_id, target_time])
                rows.append(row)

            feature_cols = [f"{feature_col}_t-{window_size - j}" for j in range(window_size)]
            all_cols = feature_cols + ["target", "start_station_id", "hour"]

            station_df = pd.DataFrame(rows, columns=all_cols)
            transformed_data.append(station_df)

        except ValueError as e:
            print(f"Skipping station_id {station_id}: {str(e)}")

    if not transformed_data:
        raise ValueError("No stations had enough data for sliding window transformation.")

    final_df = pd.concat(transformed_data, ignore_index=True)

    features_df = final_df[feature_cols + ["hour", "start_station_id"]]
    targets = final_df["target"]

    return features_df, targets


def transform_ts_data_into_features_and_target(
    df: pd.DataFrame,
    feature_col: str = "ride_count",
    window_size: int = 12,
    step_size: int = 1
) -> tuple:
    """
    Transforms Citi Bike time series data for all start_station_ids into tabular features + target,
    using a sliding window approach.

    Args:
        df (pd.DataFrame): Time series with ['hour', 'start_station_id', feature_col]
        feature_col (str): Value to predict (e.g. 'ride_count')
        window_size (int): Number of past time steps to use as features
        step_size (int): Number of rows to slide the window

    Returns:
        tuple: (features_df, targets_series)
    """
    station_ids = df["start_station_id"].unique()
    transformed_data = []

    for station_id in station_ids:
        try:
            station_df = df[df["start_station_id"] == station_id].reset_index(drop=True)
            values = station_df[feature_col].values
            times = station_df["hour"].values

            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            rows = []
            for i in range(0, len(values) - window_size, step_size):
                features = values[i : i + window_size]
                target = values[i + window_size]
                timestamp = times[i + window_size]
                row = np.append(features, [target, station_id, timestamp])
                rows.append(row)

            feature_cols = [f"{feature_col}_t-{window_size - j}" for j in range(window_size)]
            all_cols = feature_cols + ["target", "start_station_id", "hour"]
            station_tabular_df = pd.DataFrame(rows, columns=all_cols)
            transformed_data.append(station_tabular_df)

        except ValueError as e:
            print(f"Skipping station {station_id}: {e}")

    if not transformed_data:
        raise ValueError("No stations could be transformed — check data or window size.")

    final_df = pd.concat(transformed_data, ignore_index=True)
    features_df = final_df[feature_cols + ["hour", "start_station_id"]]
    targets = final_df["target"]

    return features_df, targets


def split_time_series_data(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits a time series DataFrame into training and testing sets based on a cutoff timestamp.

    Args:
        df (pd.DataFrame): Input DataFrame with a time column (e.g., 'hour' or 'pickup_hour').
        cutoff_date (datetime): The timestamp used to divide the data.
        target_column (str): The column name for the target variable.

    Returns:
        Tuple:
            X_train (pd.DataFrame): Features for training.
            y_train (pd.Series): Targets for training.
            X_test (pd.DataFrame): Features for testing.
            y_test (pd.Series): Targets for testing.
    """
    # Rename 'hour' to 'pickup_hour' if necessary for consistency
    time_col = "hour" if "hour" in df.columns else "pickup_hour"

    # Split based on the cutoff timestamp
    train_data = df[df[time_col] < cutoff_date].reset_index(drop=True)
    test_data = df[df[time_col] >= cutoff_date].reset_index(drop=True)

    # Separate features and targets
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    return X_train, y_train, X_test, y_test


# def fetch_batch_raw_data(
#     from_date: Union[datetime, str],
#     to_date: Union[datetime, str]
# ) -> pd.DataFrame:
#     """
#     Simulates production batch data by sampling Citi Bike data from 52 weeks ago.

#     Args:
#         from_date (datetime or str): Start datetime of batch window.
#         to_date (datetime or str): End datetime of batch window.

#     Returns:
#         pd.DataFrame: Simulated batch data with 'started_at' and 'start_station_id'.
#     """
#     # Parse strings if needed
#     if isinstance(from_date, str):
#         from_date = datetime.fromisoformat(from_date)
#     if isinstance(to_date, str):
#         to_date = datetime.fromisoformat(to_date)

#     # Validate input
#     if not isinstance(from_date, datetime) or not isinstance(to_date, datetime):
#         raise ValueError("from_date and to_date must be datetime or ISO strings")
#     if from_date >= to_date:
#         raise ValueError("from_date must be earlier than to_date")

#     # Shift 52 weeks back
#     hist_from = from_date - timedelta(weeks=52)
#     hist_to = to_date - timedelta(weeks=52)

#     # Load month of hist_from
#     rides_from = load_and_process_citibike_data(
#         year=hist_from.year, months=[hist_from.month]
#     )
#     #rides_from = rides_from[rides_from['started_at'] >= pd.to_datetime(hist_from)]

#     hist_from = pd.to_datetime(hist_from).tz_localize(None)
#     rides_from['started_at'] = pd.to_datetime(rides_from['started_at']).dt.tz_localize(None)
#     rides_from = rides_from[rides_from['started_at'] >= hist_from]


#     # If hist_to is a different month, load and combine
#     if hist_to.month != hist_from.month:
#         rides_to = load_and_process_citibike_data(
#             year=hist_to.year, months=[hist_to.month]
#         )
#         #rides_to = rides_to[rides_to['started_at'] < pd.to_datetime(hist_to)]
#         rides = pd.concat([rides_from, rides_to], ignore_index=True)
#     else:
#         rides = rides_from

#     # Shift forward 52 weeks
#     rides["started_at"] += timedelta(weeks=52)

#     # Sort for consistency
#     rides.sort_values(by=["start_station_id", "started_at"], inplace=True)

#     return rides

from typing import Union
from datetime import datetime, timedelta
import pandas as pd

def fetch_batch_raw_data(
    from_date: Union[datetime, str],
    to_date: Union[datetime, str]
) -> pd.DataFrame:
    """
    Simulates production batch data by sampling Citi Bike data from 52 weeks ago.

    Args:
        from_date (datetime or str): Start datetime of batch window.
        to_date (datetime or str): End datetime of batch window.

    Returns:
        pd.DataFrame: Simulated batch data with 'started_at' and 'start_station_id'.
    """
    # Parse strings if needed
    if isinstance(from_date, str):
        from_date = datetime.fromisoformat(from_date)
    if isinstance(to_date, str):
        to_date = datetime.fromisoformat(to_date)

    # Validate input
    if not isinstance(from_date, datetime) or not isinstance(to_date, datetime):
        raise ValueError("from_date and to_date must be datetime or ISO strings")
    if from_date >= to_date:
        raise ValueError("from_date must be earlier than to_date")

    # Shift 52 weeks back
    hist_from = from_date - timedelta(weeks=52)
    hist_to = to_date - timedelta(weeks=52)

    # Ensure both comparison values are tz-naive
    hist_from = pd.to_datetime(hist_from).tz_localize(None)
    hist_to = pd.to_datetime(hist_to).tz_localize(None)

    # Load month of hist_from
    rides_from = load_and_process_citibike_data(
        year=hist_from.year, months=[hist_from.month]
    )
    rides_from['started_at'] = pd.to_datetime(rides_from['started_at'], errors='coerce').dt.tz_localize(None)
    rides_from = rides_from[rides_from['started_at'] >= hist_from]

    # If hist_to is a different month, load and combine
    if hist_to.month != hist_from.month:
        rides_to = load_and_process_citibike_data(
            year=hist_to.year, months=[hist_to.month]
        )
        rides_to['started_at'] = pd.to_datetime(rides_to['started_at'], errors='coerce').dt.tz_localize(None)
        rides_to = rides_to[rides_to['started_at'] < hist_to]
        rides = pd.concat([rides_from, rides_to], ignore_index=True)
    else:
        rides = rides_from

    # Shift forward 52 weeks to simulate current batch
    rides["started_at"] += timedelta(weeks=52)

    # Sort for consistency
    rides.sort_values(by=["start_station_id", "started_at"], inplace=True)

    return rides



def transform_ts_data_into_features(
    df: pd.DataFrame,
    feature_col: str = "ride_count",
    window_size: int = 12,
    step_size: int = 1
) -> pd.DataFrame:
    """
    Transforms Citi Bike time series data into feature-only tabular format using a sliding window,
    for all unique start_station_ids. This is typically used for inference (no target).

    Args:
        df (pd.DataFrame): Time series data with 'hour', 'start_station_id', and feature_col.
        feature_col (str): Column to use as features (default: 'ride_count').
        window_size (int): Number of time steps to use in each window.
        step_size (int): Step size to slide window.

    Returns:
        pd.DataFrame: A DataFrame with feature columns, 'hour' and 'start_station_id'.
    """
    station_ids = df["start_station_id"].unique()
    transformed_data = []

    for station_id in station_ids:
        try:
            station_df = df[df["start_station_id"] == station_id].reset_index(drop=True)
            values = station_df[feature_col].values
            times = station_df["hour"].values

            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            rows = []
            for i in range(0, len(values) - window_size + 1, step_size):
                features = values[i : i + window_size]
                timestamp = times[i + window_size - 1]
                row = np.append(features, [station_id, timestamp])
                rows.append(row)

            feature_cols = [f"{feature_col}_t-{window_size - j}" for j in range(window_size)]
            all_cols = feature_cols + ["start_station_id", "hour"]

            station_feature_df = pd.DataFrame(rows, columns=all_cols)
            transformed_data.append(station_feature_df)

        except ValueError as e:
            print(f"Skipping station {station_id}: {e}")

    if not transformed_data:
        raise ValueError("No data could be transformed. Check if data is too short or window too large.")

    return pd.concat(transformed_data, ignore_index=True)
from datetime import timedelta
from typing import Optional
import pandas as pd
import plotly.express as px

def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: int,
    predictions: Optional[pd.Series] = None,
):
    """
    Plots the time series ride data for a specific Citi Bike station using a row from the tabular dataset.

    Args:
        features (pd.DataFrame): Tabular data with lagged ride_count features + metadata.
        targets (pd.Series): Series of actual ride counts (target values).
        row_id (int): Row index to visualize.
        predictions (Optional[pd.Series]): Optional predicted values for comparison.

    Returns:
        plotly.graph_objects.Figure: Interactive time series plot.
    """
    # Extract feature row and target value
    location_features = features.iloc[row_id]
    actual_target = targets.iloc[row_id]

    # Identify lag feature columns (e.g., 'ride_count_t-672' to 'ride_count_t-1')
    time_series_columns = [
        col for col in features.columns if col.startswith("ride_count_t-")
    ]
    time_series_values = [location_features[col] for col in time_series_columns] + [actual_target]

    # Compute hourly timestamps for the time series window
    time_series_dates = pd.date_range(
        start=location_features["hour"] - timedelta(hours=len(time_series_columns)),
        end=location_features["hour"],
        freq="h",
    )

    # Title with metadata
    title = f"Hour: {location_features['hour']}, Station ID: {location_features['start_station_id']}"

    # Base plot
    fig = px.line(
        x=time_series_dates,
        y=time_series_values,
        template="plotly_white",
        markers=True,
        title=title,
        labels={"x": "Time", "y": "Ride Counts"},
    )

    # Add actual target (last point) as green marker
    fig.add_scatter(
        x=[time_series_dates[-1]],
        y=[actual_target],
        line_color="green",
        mode="markers",
        marker_size=10,
        name="Actual Value",
    )

    # Optional prediction as red "X"
    if predictions is not None:
        predicted_value = predictions[row_id]
        fig.add_scatter(
            x=[time_series_dates[-1]],
            y=[predicted_value],
            line_color="red",
            mode="markers",
            marker_symbol="x",
            marker_size=15,
            name="Prediction",
        )

    return fig

from datetime import timedelta
import pandas as pd
import plotly.express as px

def plot_prediction(features: pd.DataFrame, prediction: pd.DataFrame):
    """
    Plots historical ride counts and overlays a prediction point for Citi Bike data.

    Args:
        features (pd.DataFrame): A DataFrame containing feature values and metadata
                                 for a single prediction window (1 row expected).
        prediction (pd.DataFrame): A DataFrame with 'predicted_demand' values to plot.

    Returns:
        plotly.graph_objects.Figure: Time series plot with prediction point.
    """
    # Identify lag feature columns (e.g., 'ride_count_t-*')
    time_series_columns = [
        col for col in features.columns if col.startswith("ride_count_t-")
    ]
    time_series_values = [
        features[col].iloc[0] for col in time_series_columns
    ] + prediction["predicted_demand"].to_list()

    # Use the last timestamp as the reference (current hour)
    current_hour = pd.Timestamp(features["hour"].iloc[0])

    # Generate timestamps from the window
    time_series_dates = pd.date_range(
        start=current_hour - timedelta(hours=len(time_series_columns)),
        end=current_hour,
        freq="h",
    )

    # Construct historical time series
    historical_df = pd.DataFrame({
        "datetime": time_series_dates,
        "rides": time_series_values
    })

    # Set title using consistent metadata
    title = f"Hour: {current_hour}, Station ID: {features['start_station_id'].iloc[0]}"

    # Base line plot
    fig = px.line(
        historical_df,
        x="datetime",
        y="rides",
        template="plotly_white",
        markers=True,
        title=title,
        labels={"datetime": "Time", "rides": "Ride Counts"},
    )

    # Overlay prediction marker
    fig.add_scatter(
        x=[current_hour],
        y=prediction["predicted_demand"].to_list(),
        line_color="red",
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        name="Prediction",
    )

    return fig
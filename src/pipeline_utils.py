import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

# Function to calculate the average ride_count over the last 4 weeks
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    last_4_weeks_columns = [
        f"ride_count_t-{7*24}",   # 1 week ago
        f"ride_count_t-{14*24}",  # 2 weeks ago
        f"ride_count_t-{21*24}",  # 3 weeks ago
        f"ride_count_t-{28*24}",  # 4 weeks ago
    ]

    # Check required columns exist
    for col in last_4_weeks_columns:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")

    X = X.copy()
    X["average_rides_last_4_weeks"] = X[last_4_weeks_columns].mean(axis=1)
    return X

# FunctionTransformer for avg ride feature
add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)

# Transformer to add hour of day and weekday
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["hour_of_day"] = pd.to_datetime(X_["hour"]).dt.hour
        X_["day_of_week"] = pd.to_datetime(X_["hour"]).dt.dayofweek
        return X_.drop(columns=["hour", "start_station_id"], errors="ignore")

# Instantiate transformer
add_temporal_features = TemporalFeatureEngineer()

# Complete feature + model pipeline
def get_pipeline(**hyper_params):
    """
    Returns a full pipeline with feature engineering and LGBMRegressor.

    Parameters:
        **hyper_params : dict
            Parameters to pass to LGBMRegressor.

    Returns:
        sklearn.pipeline.Pipeline
    """
    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyper_params)
    )
    return pipeline
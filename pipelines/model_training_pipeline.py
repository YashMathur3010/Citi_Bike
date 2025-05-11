import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error

import src.config as config
from src.data_utils import transform_ts_data_into_features_and_target
from src.inference import (
    fetch_days_data,
    get_hopsworks_project,
    load_metrics_from_registry,
    load_model_from_registry,
)
from src.pipeline_utils import get_pipeline

# Fetch historical data from feature store (shifted 52 weeks)
print("Fetching data from group store ...")
ts_data = fetch_days_data(180)  # fetch 180 days from last year

print("Transforming to features and targets ...")
features, targets = transform_ts_data_into_features_and_target(
    ts_data, window_size=24 * 28, step_size=23
)

# Train LightGBM pipeline
pipeline = get_pipeline()
print("Training model ...")
pipeline.fit(features, targets)

predictions = pipeline.predict(features)
test_mae = mean_absolute_error(targets, predictions)

# Compare against previously logged model
metric = load_metrics_from_registry()
print(f"The new MAE is {test_mae:.4f}")
print(f"The previous MAE is {metric['test_mae']:.4f}")

if test_mae < metric.get("test_mae", float("inf")):
    print("Logging new model to registry ...")

    # Upload updated model
    project = get_hopsworks_project()
    model_dir = config.MODEL_OUTPUT_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_dir / "lgb_model.pkl")

    model_schema = Schema(features)
    model_schema = ModelSchema(model_schema)

    mr = project.get_model_registry()
    mr.python.create_model(
        name=config.MODEL_NAME,
        model_dir=model_dir,
        metrics={"test_mae": test_mae},
        model_schema=model_schema,
        description="LightGBM model trained on Citi Bike data",
    )
else:
    print("No improvement. Skipping model registry update.")

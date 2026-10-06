import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlflow
import pandas as pd
from dotenv import load_dotenv
from evidently.future.datasets import DataDefinition, Dataset
from evidently.future.metrics import DriftedColumnsCount, MissingValueCount, ValueDrift
from evidently.future.report import Report
from prefect import flow, get_run_logger, task
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REFERENCE_DATA_PATH = DATA_DIR / "reference_data.parquet"
DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5555/ny_taxi"
MODEL_NAME = "green-taxi-ride-duration"


def preprocess(
    filename: str, num_features: list[str], cat_features: list[str], target: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load one taxi batch and keep the same modeling columns used in notebooks."""
    df = pd.read_parquet(filename)
    received_rows = len(df)

    # Duration is the supervised label; filtering keeps training and monitoring
    # aligned with the outlier rules introduced in notebook 01.
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)]
    # The model one-hot encodes the location IDs, so they must be strings.
    df[cat_features] = df[cat_features].astype(str)
    row_counts = {
        "received_rows": received_rows,
        "retained_rows": len(df),
        "rejected_rows": received_rows - len(df),
    }
    return df[cat_features + num_features + [target]], row_counts


def load_model(
    model_name: str, mlflow_tracking_uri: str
) -> tuple[mlflow.pyfunc.PyFuncModel, int]:
    """Load the model version that currently holds the production alias."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    version = int(
        mlflow.MlflowClient()
        .get_model_version_by_alias(model_name, "production")
        .version
    )
    # Pin the resolved version even if the alias changes during this run.
    return mlflow.pyfunc.load_model(f"models:/{model_name}/{version}"), version


def prepare_evidently_dataset(
    df: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    pred_col: str,
) -> Dataset:
    """Attach column roles so Evidently treats features and predictions correctly."""
    data_definition = DataDefinition(
        numerical_columns=num_features + [pred_col],
        categorical_columns=cat_features,
    )
    return Dataset.from_pandas(df, data_definition)


def create_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    pred_col: str = "prediction",
) -> dict:
    """Compare the saved reference snapshot with the current scored batch."""
    reference_dataset = prepare_evidently_dataset(
        reference_data, num_features, cat_features, pred_col
    )
    current_dataset = prepare_evidently_dataset(
        current_data, num_features, cat_features, pred_col
    )

    report_metrics: list[Any] = [
        cast(Any, ValueDrift)(column=pred_col),
        cast(Any, DriftedColumnsCount)(),
        cast(Any, MissingValueCount)(column=pred_col),
    ]

    report = Report(
        metrics=report_metrics,
        include_tests=True,
    )

    return report.run(
        reference_data=reference_dataset, current_data=current_dataset
    ).dict()


def load_report(
    result: dict, database_url: str, batch_month: datetime, run_metrics: dict
) -> None:
    """Persist the small metric set that powers the Grafana dashboard."""
    logger = get_run_logger()
    logger.info("Loading results to database")
    engine = create_engine(database_url)

    report_dict = {
        # Plot each run at the month of data it scored, not at the time it ran.
        "timestamp": batch_month,
        "prediction_drift": result["metrics"][0]["value"],
        "num_drifted_columns": result["metrics"][1]["value"]["count"],
        "share_missing_values": result["metrics"][2]["value"]["share"],
        **run_metrics,
    }

    df_report = pd.DataFrame([report_dict])
    logger.info("Writing %s row to drift_metrics", len(df_report))
    try:
        with engine.begin() as connection:
            # Existing lesson databases retain their rows when new metrics arrive.
            if inspect(connection).has_table("drift_metrics"):
                connection.execute(
                    text(
                        """ALTER TABLE drift_metrics
                        ADD COLUMN IF NOT EXISTS mae DOUBLE PRECISION,
                        ADD COLUMN IF NOT EXISTS received_rows BIGINT,
                        ADD COLUMN IF NOT EXISTS retained_rows BIGINT,
                        ADD COLUMN IF NOT EXISTS rejected_rows BIGINT,
                        ADD COLUMN IF NOT EXISTS model_name TEXT,
                        ADD COLUMN IF NOT EXISTS model_version BIGINT,
                        ADD COLUMN IF NOT EXISTS run_timestamp TIMESTAMP WITH TIME ZONE
                        """
                    )
                )
            df_report.to_sql(
                name="drift_metrics", con=connection, if_exists="append", index=False
            )
    finally:
        engine.dispose()


@task
def apply_model(
    filename: str,
    batch_month: datetime,
    model_name: str,
    mlflow_tracking_uri: str,
    database_url: str,
) -> None:
    logger = get_run_logger()
    run_timestamp = datetime.now(UTC)
    logger.info("Reading current batch from %s", filename)

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {REFERENCE_DATA_PATH}. "
            "Run the final cell from 03-intro-to-evidently.ipynb first."
        )

    # Notebook 03 writes this file with the training split.
    reference_data = pd.read_parquet(REFERENCE_DATA_PATH)

    cat_features = ["PULocationID", "DOLocationID"]
    num_features = ["trip_distance"]
    target = "duration"

    current_data, row_counts = preprocess(filename, num_features, cat_features, target)
    logger.info(
        "Batch rows: received=%s, retained=%s, rejected=%s",
        row_counts["received_rows"],
        row_counts["retained_rows"],
        row_counts["rejected_rows"],
    )
    if current_data.empty:
        raise ValueError(
            "No rows remain after filtering this batch; check the input data."
        )

    logger.info("Loading production model %s", model_name)
    loaded_model, model_version = load_model(model_name, mlflow_tracking_uri)
    logger.info("Using model %s version %s", model_name, model_version)

    # Score the reference and the current batch with the same model, so prediction
    # drift reflects changes in the data and not a newly promoted model.
    for df in (reference_data, current_data):
        df["prediction"] = loaded_model.predict(
            df[cat_features + num_features].to_dict(orient="records")
        )

    logger.info("Creating Evidently report")
    result = create_report(
        reference_data, current_data, num_features, cat_features, pred_col="prediction"
    )
    run_metrics = {
        "mae": float(
            mean_absolute_error(current_data[target], current_data["prediction"])
        ),
        **row_counts,
        "model_name": model_name,
        "model_version": model_version,
        "run_timestamp": run_timestamp,
    }
    load_report(result, database_url, batch_month, run_metrics)


@flow(name="Predict Green Taxi Trip Duration")
def run(
    mlflow_tracking_uri: str,
    model_name: str = MODEL_NAME,
    year: int = 2025,
    month: int = 2,
    database_url: str = DEFAULT_DATABASE_URL,
) -> None:
    logger = get_run_logger()
    logger.info("Running flow for %s-%02d", year, month)

    # Pandas can stream this Parquet file directly from the public NYC TLC bucket.
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    batch_month = datetime(year, month, 1, tzinfo=UTC)
    apply_model(filename, batch_month, model_name, mlflow_tracking_uri, database_url)


if __name__ == "__main__":
    load_dotenv()

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    if not mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is missing. Add it to your .env file.")

    run(
        mlflow_tracking_uri=mlflow_tracking_uri,
        year=2025,
        month=2,
        database_url=database_url,
    )

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
from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REFERENCE_DATA_PATH = DATA_DIR / "reference_data.parquet"
DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5555/ny_taxi"


def preprocess(
    filename: str, num_features: list[str], cat_features: list[str], target: str
) -> pd.DataFrame:
    """Load one taxi batch and keep the same modeling columns used in notebooks."""
    df = pd.read_parquet(filename)

    # Duration is the supervised label; filtering keeps training and monitoring
    # aligned with the outlier rules introduced in notebook 01.
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)]
    # The model one-hot encodes the location IDs, so they must be strings.
    df[cat_features] = df[cat_features].astype(str)
    return df[cat_features + num_features + [target]]


def load_model(run_id: str, mlflow_tracking_uri: str):
    """Load the exact model run selected by the RUN_ID setting."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logged_model = f"runs:/{run_id}/mlflow-model"
    return mlflow.pyfunc.load_model(logged_model)


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
        cast(Any, ValueDrift)(column="prediction"),
        cast(Any, DriftedColumnsCount)(),
        cast(Any, MissingValueCount)(column="prediction"),
    ]

    report = Report(
        metrics=report_metrics,
        include_tests=True,
    )

    return report.run(
        reference_data=reference_dataset, current_data=current_dataset
    ).dict()


def load_report(result: dict, database_url: str) -> None:
    """Persist the small metric set that powers the Grafana dashboard."""
    logger = get_run_logger()
    logger.info("Loading results to database")
    engine = create_engine(database_url)

    report_dict = {
        "timestamp": datetime.now(UTC),
        "prediction_drift": result["metrics"][0]["value"],
        "num_drifted_columns": result["metrics"][1]["value"]["count"],
        "share_missing_values": result["metrics"][2]["value"]["share"],
    }

    df_report = pd.DataFrame([report_dict])
    logger.info("Writing %s row to drift_metrics", len(df_report))
    df_report.to_sql(name="drift_metrics", con=engine, if_exists="append", index=False)


@task
def apply_model(
    filename: str, run_id: str, mlflow_tracking_uri: str, database_url: str
) -> None:
    logger = get_run_logger()
    logger.info("Reading current batch from %s", filename)

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {REFERENCE_DATA_PATH}. "
            "Run the final cell from 03-intro-to-evidently.ipynb first."
        )

    # Notebook 03 writes this file after scoring the reference split.
    reference_data = pd.read_parquet(REFERENCE_DATA_PATH)

    cat_features = ["PULocationID", "DOLocationID"]
    num_features = ["trip_distance"]
    target = "duration"

    current_data = preprocess(filename, num_features, cat_features, target)

    logger.info("Loading model run %s", run_id)
    loaded_model = load_model(run_id, mlflow_tracking_uri)
    current_data["prediction"] = loaded_model.predict(
        current_data[cat_features + num_features].to_dict(orient="records")
    )

    logger.info("Creating Evidently report")
    result = create_report(
        reference_data, current_data, num_features, cat_features, pred_col="prediction"
    )
    load_report(result, database_url)


@flow(name="Predict Green Taxi Trip Duration")
def run(
    run_id: str,
    mlflow_tracking_uri: str,
    year: int = 2025,
    month: int = 2,
    database_url: str = DEFAULT_DATABASE_URL,
) -> None:
    logger = get_run_logger()
    logger.info("Running flow for %s-%02d", year, month)

    # Pandas can stream this Parquet file directly from the public NYC TLC bucket.
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    apply_model(filename, run_id, mlflow_tracking_uri, database_url)


if __name__ == "__main__":
    load_dotenv()

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    run_id = os.getenv("RUN_ID")
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    if not mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is missing. Add it to your .env file.")
    if not run_id:
        raise ValueError(
            "RUN_ID is missing. Add it to your .env file after notebook 01."
        )

    run(
        run_id=run_id,
        mlflow_tracking_uri=mlflow_tracking_uri,
        year=2025,
        month=2,
        database_url=database_url,
    )

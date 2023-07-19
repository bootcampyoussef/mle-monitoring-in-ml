import os
import uuid
import pandas as pd
import mlflow
from prefect import task, flow, get_run_logger
from datetime import datetime
from dotenv import load_dotenv


from sqlalchemy import create_engine

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)


def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


def preprocess(filename, num_features, cat_features, target):
    df = pd.read_parquet(filename)
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] < 8)]
    df = df[cat_features + num_features + [target]]
    df["ride_id"] = generate_uuids(len(df))
    return df


def load_model(run_id, MLFLOW_TRACKING_URI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logged_model = f"runs:/{run_id}/model"
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model


def create_report(reference_data, df, num_features, cat_features, target):
    logger = get_run_logger()
    logger.info(f"Creating report...")

    column_mapping = ColumnMapping(
        prediction="prediction",
        numerical_features=num_features,
        categorical_features=cat_features,
        target=target,
    )

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="prediction"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
    )

    report.run(
        reference_data=reference_data, current_data=df, column_mapping=column_mapping
    )
    result = report.as_dict()

    return result


def load_report(result):
    logger = get_run_logger()
    logger.info(f"Loading results to database...")
    engine = create_engine("postgresql://postgres:postgres@localhost:5432/ny_taxi")

    report_dict = {
        "timestamp": datetime.now(),
        "prediction_drift": result["metrics"][0]["result"]["drift_score"],
        "num_drifted_columns": result["metrics"][1]["result"][
            "number_of_drifted_columns"
        ],
        "share_missing_values": result["metrics"][2]["result"]["current"][
            "share_of_missing_values"
        ],
    }

    df_report = pd.DataFrame([report_dict])

    logger.info(f"Dataframe: {df_report.head()}")

    df_report.to_sql("drift_metrics", engine, if_exists="append", index=False)


@task
def apply_model(filename, run_id, MLFLOW_TRACKING_URI):
    logger = get_run_logger()
    logger.info(f"Reading data from {filename}")

    # load the reference dataset
    reference_data = pd.read_parquet("./data/reference_data.parquet")

    cat_features = ["PULocationID", "DOLocationID"]
    num_features = ["trip_distance", "passenger_count", "fare_amount", "total_amount"]
    target = "duration"

    df = preprocess(filename, num_features, cat_features, target)

    logger.info(f"Loading the model {run_id}")
    loaded_model = load_model(run_id, MLFLOW_TRACKING_URI)
    y_pred = loaded_model.predict(df[cat_features + num_features])

    df["prediction"] = y_pred

    result = create_report(reference_data, df, num_features, cat_features, target)
    load_report(result)


@flow(name="Predict Green Taxi Trip Duration")
def run(
    run_id: str,
    google_sa_key: str,
    MLFLOW_TRACKING_URI,
    year: int = 2021,
    month: int = 2,
):
    logger = get_run_logger()
    logger.info(f"Running flow for {year}-{month:02d}")
    path = os.path.dirname(__file__)
    logger.info(f"Current path: {path}")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_sa_key

    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    run_id = run_id
    apply_model(filename, run_id, MLFLOW_TRACKING_URI)


if __name__ == "__main__":
    load_dotenv()

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    RUN_ID = os.getenv("RUN_ID")
    GOOGLE_SA_KEY = os.getenv("SA_KEY")

    run(run_id=RUN_ID, google_sa_key=GOOGLE_SA_KEY, MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI)


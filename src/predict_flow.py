import os
import uuid
import pandas as pd
import mlflow
from prefect import task, flow, get_run_logger
from datetime import datetime
from dotenv import load_dotenv
from typing import List


from sqlalchemy import create_engine

from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount
from evidently.metrics.column_statistics import MinValue, MaxValue, MeanValue
from evidently.presets import DataDriftPreset, DataSummaryPreset


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
    #df["ride_id"] = generate_uuids(len(df))
    return df


def load_model(run_id, mlflow_tracking_uri):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logged_model = f"runs:/{run_id}/mlflow-model"
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model


def prepare_evidently_dataset(df: pd.DataFrame, num_features: List[str], cat_features: List[str], pred_col: str) -> Dataset:
    """Prepare an Evidently Dataset from a pandas DataFrame."""
    data_definition = DataDefinition(numerical_columns=num_features + [pred_col], categorical_columns=cat_features)
    evidently_dataset = Dataset.from_pandas(df, data_definition)
    return evidently_dataset

def create_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, num_features: List[str] , cat_features: List[str], pred_col: str = 'prediction') -> dict:
    """Create an Evidently report comparing reference and current datasets."""
    logger = get_run_logger()
    logger.info(f"Creating report...")

    reference_dataset = prepare_evidently_dataset(reference_data, num_features, cat_features, pred_col)
    current_dataset = prepare_evidently_dataset(current_data, num_features, cat_features, pred_col)

    report = Report(metrics=[
        ValueDrift(column='prediction'), 
        DriftedColumnsCount(),
        MissingValueCount(column='prediction')], include_tests=True)

    result = report.run(reference_data=reference_dataset, current_data=current_dataset).dict()

    return result


def load_report(result):
    """Load the report into the Postgres database."""
    logger = get_run_logger()
    logger.info(f"Loading results to database...")
    engine = create_engine("postgresql://postgres:postgres@localhost:5555/ny_taxi")

    report_dict = {
        "timestamp": datetime.now(),
        "prediction_drift": result['metrics'][0]['value'],
        "num_drifted_columns": result['metrics'][1]['value']['count'],
        "share_missing_values": result['metrics'][2]['value']['share']}

    df_report = pd.DataFrame([report_dict])

    logger.info(f"Dataframe: {df_report.head()}")

    df_report.to_sql(name="drift_metrics", con=engine, if_exists="append", index=False)


@task
def apply_model(filename, run_id, mlflow_tracking_uri):
    logger = get_run_logger()
    logger.info(f"Reading data from {filename}")

    # load the reference dataset
    logger.info(f"Loading reference data")
    reference_data = pd.read_parquet("./data/reference_data.parquet")

    # define features
    cat_features = ["PULocationID", "DOLocationID"]
    num_features = ["trip_distance", "passenger_count", "fare_amount", "total_amount"]
    # define target
    target = "duration"

    # preprocess current data
    current_data = preprocess(filename, num_features, cat_features, target)

    logger.info(f"Loading the model {run_id}")
    loaded_model = load_model(run_id, mlflow_tracking_uri)
    logger.info(f"Applying the model to make predictions")
    y_pred = loaded_model.predict(current_data[cat_features + num_features])

    # add predictions to current data
    current_data["prediction"] = y_pred

    # create evidently report
    logger.info(f"Creating evidently report")
    result = create_report(reference_data, current_data, num_features, cat_features, pred_col='prediction')

    # load report to database
    logger.info(f"Report result: {result}")
    load_report(result)


@flow(name="Predict Green Taxi Trip Duration")
def run(
    run_id: str,
    gcp_key: str,
    mlflow_tracking_uri,
    year: int = 2025,
    month: int = 2,
):
    logger = get_run_logger()
    logger.info(f"Running flow for {year}-{month:02d}")
    path = os.path.dirname(__file__)
    logger.info(f"Current path: {path}")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_key

    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    run_id = run_id
    apply_model(filename, run_id, mlflow_tracking_uri)


if __name__ == "__main__":
    load_dotenv()

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    RUN_ID = os.getenv("RUN_ID")
    gcp_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    run(run_id=RUN_ID, gcp_key=gcp_key, mlflow_tracking_uri=MLFLOW_TRACKING_URI, year=2025, month=2)


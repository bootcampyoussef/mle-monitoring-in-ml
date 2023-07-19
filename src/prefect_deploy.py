from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from predict_flow import run
from dotenv import load_dotenv
import os

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
RUN_ID = os.getenv("RUN_ID")
GOOGLE_SA_KEY = os.getenv("SA_KEY")


deployment = Deployment.build_from_flow(
    flow=run,
    name="ride_duration_prediction_monitoring",
    parameters={
        "run_id": RUN_ID,
        "google_sa_key": GOOGLE_SA_KEY,
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    },
    tags=["batch", "predict", "prefect"],
)

deployment.apply()

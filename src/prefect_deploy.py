import os
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv
from prefect.client.schemas.schedules import RRuleSchedule

from predict_flow import run

# Load local connection settings used by the scheduled flow parameters below.
load_dotenv()

# Start the first run shortly after launch so you can see the deployment run
# without waiting for a real monthly schedule tick.
now = datetime.now(UTC) + timedelta(minutes=2)
rrule_string = (
    f"DTSTART:{now.strftime('%Y%m%dT%H%M%SZ')}\n"
    f"FREQ=MONTHLY;INTERVAL=2;BYMONTHDAY={now.day};BYHOUR={now.hour};BYMINUTE={now.minute}"
)

schedule_rule = RRuleSchedule(rrule=rrule_string)

if __name__ == "__main__":
    # `serve` registers the flow with the local Prefect API and keeps a lightweight
    # worker process alive to poll for scheduled or manually triggered runs.
    run.serve(
        name="ride_duration_prediction_monitoring",
        schedules=[schedule_rule],
        parameters={
            "mlflow_tracking_uri": os.getenv(
                "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
            ),
            "database_url": os.getenv(
                "DATABASE_URL",
                "postgresql+psycopg://postgres:postgres@localhost:5555/ny_taxi",
            ),
            "year": 2025,
            "month": 2,
        },
        tags=["batch", "predict", "prefect"],
        # Run one flow at a time so simultaneous runs never write to Postgres together.
        limit=1,
    )

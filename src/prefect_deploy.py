import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from predict_flow import run
from prefect.client.schemas.schedules import RRuleSchedule, CronSchedule 


# Load environment variables from a .env file
load_dotenv()

# Start the first run 2 minutes from now, then every 2 months on the same day and time
now = datetime.now(timezone.utc) + timedelta(minutes=2)
rrule_string = (
    f"DTSTART:{now.strftime('%Y%m%dT%H%M%SZ')}\n"
    f"FREQ=MONTHLY;INTERVAL=2;BYMONTHDAY={now.day};BYHOUR={now.hour};BYMINUTE={now.minute}"
)

schedule_rule = RRuleSchedule(rrule=rrule_string)

if __name__ == "__main__":
    # Start the flow with a schedule
    run.serve(
        name="ride_duration_prediction",
        schedules=[
            schedule_rule,
            ],
        parameters={
            "run_id": os.getenv("RUN_ID"),
            "gcp_key": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
            "year":2025,
            "month":2,
        },
        tags=["batch", "predict", "prefect"],
    )



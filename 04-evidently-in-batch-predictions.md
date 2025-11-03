# Evidently in Batch Predictions

For this example we will use prefect to schedule a prediction job for a model that was trained in the previous example and load the evidently report data into a postgres database. The first thing we will do is spin up a postgres database and a grafana instance using docker-compose.

Grafana is a dashboarding tool that we will use to visualize the data in our database. 


```yml
services:
  db_service:
    image: postgres:latest
    container_name: postgres_db
    restart: always
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: ny_taxi
    ports:
      - "5555:5432"
    networks:
      - monitoring-net

  grafana_service:
    image: grafana/grafana:latest
    container_name: grafana
    depends_on:
      - db_service 
    ports:
      - "3000:3000"
    restart: unless-stopped
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/etc/grafana/dashboards
      - grafana-data:/var/lib/grafana
    networks:
      - monitoring-net

volumes:
  dbdata:
  grafana-data:
networks:
  monitoring-net:
    driver: bridge
```

And then we can start it up with docker-compose:

```bash
docker-compose up -d
```

The grafana settings are stored in the `grafana` directory. We set the data source to the postgres database. And created a dashboard for the data drift in `grafana/data_drift.json`.

```yml
# config file version
apiVersion: 1

# list of datasources to insert/update
# available in the database
datasources:
  - name: PostgreSQL
    type: postgres
    url: db_service:5432
    user: postgres
    secureJsonData:
      password: 'postgres'
    jsonData:
      database: ny_taxi
      sslmode: 'disable'
    isDefault: true
```

Once started you can access the grafana dashboard at http://localhost:3000. The default username and password are both `admin`. You will be prompted to change the password you can skip that. You should be able the see the dashboard but of course without any data because we haven't loaded any yet.


Now let's create a prefect flow that will make the batch predcitions and load the evidently report into the database. 

You can find the code in `src/predict_flow.py`. The flow will create an evidently report with this code:

```python
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
```

And then we will load the data into the database with this code:

```python
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
```

You will need a `reference_data.parquet` file to run it. You can create your reference data with the code provided in 03-intro-to-evidently.ipynb and store it as a parquet file.


## **Scheduling Predictions with Prefect**

In this section, we will **schedule the batch predictions** using **Prefect** — a modern **workflow orchestration system** that makes it easy to **automate, schedule, and monitor** your data pipelines.

Prefect allows you to define your Python workflows (called *flows*) and then execute them automatically on a schedule, either **locally** or using **Prefect Cloud**.

You can learn more from the official [Prefect documentation](https://docs.prefect.io/v3/) and review the **Workflow Orchestration** materials from the previous week for background.

### **Step 1 — Configure Prefect to Use the Local Server**
Once the Prefect server is running, Prefect needs to know **where** to send API requests.  
You’ll do this by setting the environment variable `PREFECT_API_URL` so that your local Prefect client connects to the running server.

Open **another terminal window** (keep the server running in the first one and make sure to be in the same Python environment where Prefect is installed), and run:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```
This tells Prefect to use the server’s API, which by default runs at `http://127.0.0.1:4200/api`.

To make sure the configuration was set correctly, you can run in the same terminal:

#### Test the configuration
```bash
prefect config view
```

You should see:
```bash
PREFECT_PROFILE='local'
PREFECT_API_URL='http://127.0.0.1:4200/api'
```
#### Test the connection to the API

You can also check that the Prefect API is live by visiting:
[http://127.0.0.1:4200/docs](http://127.0.0.1:4200/docs).
This opens the Prefect interactive API documentation (Swagger UI), which confirms your local server is running and accessible. Please feel free to explore the available endpoints.

---

### **Step 2 — Create a Prefect Deployment**

Now that our Prefect flow is defined in `prefect_flow.py`, it’s time to make it **deployable and schedulable**.  

In Prefect, a **deployment** is a specific configuration of a flow that includes:
- Its **schedule** (when it should run)
- Any **default parameters**
- Optional **tags** and **metadata**

Deployments let you run your flow automatically — either on a **regular schedule** or in response to **specific events**.

#### 2.1 Create a Deployment Script
We’ll create a new Python file called `prefect_deploy.py` in the same folder as `prefect_flow.py`.  
This script will:
1. Import your Prefect flow  
2. Load environment variables from `.env`  
3. Define a schedule using an [RFC 5545 RRule](https://datatracker.ietf.org/doc/html/rfc5545)  
4. Serve the flow deployment locally using `run.serve()` so Prefect can execute it automatically

Here’s the complete code:

```python
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
        name="ride_duration_prediction_monitoring",
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
```
**What this Schedule does**:
- Starts the first flow run **two minutes after you execute the script**,so you can test it immediately.
- Then repeats **every two months**, on the **same day and time** as that first run.
- You can modify the rule to any schedule you like.

For example, to run the flow **every month on the 2nd at 3:00am**, you could use the following `rrule_string`  in the script:

```python
# Run on the 2nd day of every month at 3:00 AM
rrule_string = "FREQ=MONTHLY;BYMONTHDAY=2;BYHOUR=3;BYMINUTE=0"
```


**What the `run.serve()` function does behind the scenes is**:

1. **Registers your flow** with the Prefect API or server.  
   - You’ll see it appear under **Deployments** in the Prefect UI.  
2. **Creates and applies the schedule** you defined (e.g., via `RRuleSchedule` or `CronSchedule`).  
3. **Starts a lightweight background process** that continuously polls the Prefect server for upcoming scheduled runs.  
4. **Passes the parameters** you specified to each flow run.

---

#### **2.2 — Run the Deployment**

Now you can run the deployment script to **register your flow** and **start the schedule**.

Open a **new terminal window** — keep your Prefect server running in the first one —  
and make sure you’ve **activated the same Python environment** where Prefect is installed.

Then, run one of the following commands depending on your current directory:

If you are in the **project root**:
```bash
python src/prefect_deploy.py
```
Once executed , Prefect will:
1. Register your flow as a deployment with the Prefect server
2. Start serving and polling for scheduled runs
3. Display a confirmation message in your terminal
   

You should see something like this — and the terminal window will remain open, continuously polling for upcoming scheduled runs:
```bash
Your flow 'Predict Green Taxi Trip Duration' is being served and polling for scheduled runs!

To trigger a run for this flow, use the following command:

        $ prefect deployment run 'Predict Green Taxi Trip Duration/ride_duration_prediction_monitoring'

You can also run your flow via the Prefect UI: http://127.0.0.1:4200/deployments/deployment/deployment_id_here
```

<div style="
  display:block;
  max-width:80%;
  padding:0.75rem 1rem;
  border-left:4px solid #0957de;
  background:#f5f9ff;
  color:#093170;
  border-radius:4px;
  font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  font-size:0.95rem;
  line-height:1.5;
  text-indent:-2.2rem;
  padding-left:3rem;">
  💡 <strong style="color:#0957de;">Note:</strong> The first run of the deployment is scheduled to start within about two minutes after you execute the script. 
</div>

You can follow the link provided in the terminal output if you want to run the flow directly via the Prefect UI, then click the <strong>Run ▶️</strong> buttun (top right). ![](./images/run_different_month.png). Repeat this a few times with different months and you should see the data in the database and the grafana dashboard should start to show some data. Finally use the year `2025` and month `5` to see the data drift.

## **Next Steps**

Retrain the model with the new data and deploy it again. You should see that the data drift is gone.
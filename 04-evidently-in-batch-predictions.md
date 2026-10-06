# Evidently in Batch Predictions

This chapter turns the earlier notebook work into a small local monitoring system. We will:

1. start a local Postgres and Grafana stack,
2. run a Prefect flow that loads a registered MLflow model,
3. compare fresh batch data to the saved reference dataset,
4. store the resulting drift metrics in Postgres,
5. inspect the monitoring dashboard in Grafana.

```mermaid
flowchart LR
    A["Monthly<br>taxi batch"] --> B["Prefect<br>flow"]
    B --> C["Load MLflow<br>model"]
    B --> D["Load<br>reference_data.parquet"]
    C --> E["Generate predictions"]
    D --> F["Evidently<br>report"]
    E --> F
    F --> G["drift_metrics table<br>in Postgres"]
    G --> H["Grafana<br>dashboard"]
```

## Prerequisites

Before starting this chapter, make sure you have already:

- copied `.env.example` to `.env`,
- completed [01-train-ml-model.ipynb](./01-train-ml-model.ipynb), which registers the model under the `production` alias,
- completed [03-intro-to-evidently.ipynb](./03-intro-to-evidently.ipynb) and created `data/reference_data.parquet`.

Your `.env` file should look like this:

```dotenv
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
PREFECT_API_URL=http://127.0.0.1:4200/api
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5555/ny_taxi
```

## Step 1: Start the local monitoring stack

The repository includes a Docker Compose file for Postgres and Grafana. The excerpt below shows the services you interact with most directly; the full file also defines persistent Docker volumes and a shared network:

```yaml
services:
  db_service:
    image: postgres:17
    container_name: postgres_db
    restart: always
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: ny_taxi
    ports:
      - "5555:5432"

  grafana_service:
    image: grafana/grafana:13.2.3
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
```

Open a new terminal in the repository root and start the services:

```bash
docker compose up -d
```

Open Grafana at [http://127.0.0.1:3000](http://127.0.0.1:3000). The default username and password are both `admin`. Grafana may ask you to change the password on first login; for a local lesson environment, you can skip that prompt.

The Grafana provisioning files are already included:

- [grafana/provisioning/datasources/postgres_ds.yaml](./grafana/provisioning/datasources/postgres_ds.yaml)
- [grafana/provisioning/dashboards/grafana_dashboards.yaml](./grafana/provisioning/dashboards/grafana_dashboards.yaml)
- [grafana/dashboards/data_drift.json](./grafana/dashboards/data_drift.json)

At this point the dashboard loads, but it will stay empty until the Prefect flow inserts rows into the `drift_metrics` table.

## Step 2: Understand the batch monitoring flow

The batch monitoring logic lives in [src/predict_flow.py](./src/predict_flow.py).

The flow does four important things:

1. Reads the current month of taxi rides.
2. Loads the registered model from MLflow.
3. Compares reference data and current predictions with Evidently, and measures duration MAE on the labeled batch.
4. Writes a small metrics record into Postgres.

The database insert is intentionally narrow so the dashboard stays easy to read:

```python
report_dict = {
    "timestamp": batch_month,
    "prediction_drift": result["metrics"][0]["value"],
    "num_drifted_columns": result["metrics"][1]["value"]["count"],
    "share_missing_values": result["metrics"][2]["value"]["share"],
    **run_metrics,
}
```

The drift metrics answer the first monitoring questions:

- Are predictions drifting?
- How many columns are drifting?
- Are missing values becoming a problem?

`run_metrics` adds the mean absolute error (`mae`, in minutes), the counts of `received_rows`, `retained_rows`, and `rejected_rows`, plus `model_name`, `model_version`, and `run_timestamp`. Counts are captured before and after the duration/distance filters, so discarded rows remain visible. MAE is evaluated only on retained trips whose actual duration is already known; this remains an offline, historical monitoring example.

The flow resolves the `production` alias once and loads that exact model version. Existing databases receive the new columns automatically without losing earlier records; those older records have no values for the new fields.

## Step 3: Start a local Prefect server

From the repository root, start the local Prefect server:

```bash
uv run prefect server start
```

In another terminal, point the Prefect client at that server:

```bash
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
uv run prefect config view
```

You should see `PREFECT_API_URL='http://127.0.0.1:4200/api'`.

You can also confirm that the API is reachable at [http://127.0.0.1:4200/docs](http://127.0.0.1:4200/docs) and the dashboard is reachable at <http://127.0.0.1:4200>.

## Step 4: Create the deployment

The deployment helper lives in [src/prefect_deploy.py](./src/prefect_deploy.py). It schedules the flow to start two minutes after launch, then repeat every two months.

Run it from the repository root:

```bash
uv run src/prefect_deploy.py
```

Behind the scenes, `run.serve(...)`:

- registers the flow with the local Prefect API,
- applies the schedule,
- starts a lightweight worker process that keeps polling for scheduled runs,
- passes your `.env` values into each run.

## Step 5: Trigger a few monitoring runs

Once the deployment is visible in the Prefect UI, trigger a few runs manually so you can compare different months.

Suggested checks:

1. Run `2025-02` first to populate the dashboard. This month looks very similar to the training data, so expect almost no drift.
2. Run `2025-05` to see a modest change.
3. Run `2025-08` and compare how much more the drift metrics move.

When you create a manual run in the Prefect UI, open the run parameters and change only `year` and `month`. Keep `model_name`, `mlflow_tracking_uri` and `database_url` as they are.

You can also keep the scheduled deployment and wait for the first run to start automatically.

If you prefer running the flow directly from Python for a quick smoke test, use:

```bash
uv run src/predict_flow.py
```

## Step 6: Explore the dashboard

After one or more runs complete, refresh Grafana and open the provisioned dashboard. The key panels summarize:

- prediction drift over time,
- number of drifted columns per batch,
- share of missing prediction values, displayed as a percentage,
- model MAE in minutes,
- a run-history table with the received, retained, and rejected row counts and the exact model version used.

The prediction-drift panel includes a line at `0.1`, the default threshold used in this lesson. It is not a universal production limit. Use MAE to check whether prediction accuracy changes: drift alone does not demonstrate worse performance.

Each point sits at the month of data it scored, not at the time the run happened, so the panels read as a timeline whatever order you trigger the runs in. Running the same month again adds a second point at the same date. The run-history table distinguishes those runs by their execution timestamp and model version.

That view is useful because a single batch rarely tells the whole story. Monitoring becomes more interesting once you can compare several runs and ask whether the signal is noisy, stable, or worsening.

The next notebook shows how to use those monitoring signals to retrain a candidate model and promote it only if it wins on a fresh labeled batch.

## Troubleshooting

### Grafana is empty

Check that:

- the Prefect run completed successfully,
- `data/reference_data.parquet` exists,
- notebook 01 registered a model with the `production` alias,
- Postgres is reachable on `localhost:5555`.

### Prefect cannot connect to the API

Restart the local server:

```bash
uv run prefect server start
```

Then, in another terminal, reapply the config:

```bash
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

### The flow cannot find the MLflow run

Verify that `MLFLOW_TRACKING_URI` points to the same tracking store you used in notebook 01. If you trained the model with a local store, keep using that same local path or server URI here.

### The flow cannot find the reference dataset

Re-run the final cell from [03-intro-to-evidently.ipynb](./03-intro-to-evidently.ipynb), which saves `data/reference_data.parquet`.

## What you built

By the end of this chapter, you have a compact local monitoring loop:

- MLflow keeps the model artifact reachable.
- Prefect schedules recurring scoring runs.
- Evidently computes monitoring signals.
- Postgres stores those signals.
- Grafana turns them into a dashboard you can inspect over time.

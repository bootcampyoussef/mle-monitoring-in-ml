# Evidently in Batch Predictions

For this example we will use prefect to schedule a prediction job for a model that was trained in the previous example and load the evidently report data into a postgres database. The first thing we will do is spin up a postgres database and a grafana instance using docker-compose.

Grafana we saw yesterday, it is a dashboarding tool that we will use to visualize the data in our database. 


```yml
version: '3.7'

services:
  db:
    image: postgres
    restart: always
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: ny_taxi
    ports:
      - "5432:5432"

  grafana:
      image: grafana/grafana
      ports:
          - 3000:3000
      restart: unless-stopped
      volumes:
          - ./grafana:/etc/grafana/
          - grafana-data:/var/lib/grafana

volumes:
    dbdata:
    prometheus-data:
    grafana-data:
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
    access: proxy
    url: db.:5432
    database: ny_taxi
    user: postgres
    secureJsonData:
      password: 'postgres'
    jsonData:
      sslmode: 'disable'
```

Once started you can access the grafana dashboard at http://localhost:3000. The default username and password are both `admin`. You will be prompted to change the password you can skip that. You should be able the see the dashboard but of course without any data because we haven't loaded any yet.


Now let's create a prefect flow that will make the batch predcitions and load the evidently report into the database. 

You can find the code in `src/predict_flow.py`. The flow will create an evidently report with this code:

```python
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
```

And then we will load the data into the database with this code:

```python
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
```

You will need a `reference_data.parquet` file to run it. You can create your reference data with the code provided in 03-intro-to-evidently.ipynb and store it as a parquet file.

To run the flow start the prefect server:

```bash
prefect server start
```

and the prefect agent:

```bash
prefect agent start -q 'default'
```

Then you need to create the deployment:

```bash
python src/prefect_deploy.py
```

Now you can run the flow via the [Prefect UI](http://127.0.0.1:4200/). In the UI you need to navigate to `Deployments` click on the `ride_duration_prediction_monitoring` deployment and then click on `Run` and than start a `Quick Run`. Repeat this a few times with different months and you should see the data in the database and the grafana dashboard should start to show some data. Finally use the year `2022` and month `2` to see the data drift.

## Next Steps

Retrain the model with the new data and deploy it again. You should see that the data drift is gone.
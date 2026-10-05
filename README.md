# Monitoring in ML

In this repository, you will build a practical monitoring workflow for a regression model trained on the [NYC Green Taxi dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). You start by training and tracking a model with MLflow, then inspect drift manually and use Evidently to automate the checks. Finally, you wire the results into a local monitoring stack built with Prefect, Postgres and Grafana, and retrain the model when drift signals justify it.

## Learning Objectives

By the end of this repository, you should be able to:

- Train and track a regression model with MLflow.
- Distinguish data drift from prediction drift.
- Build and interpret an Evidently monitoring report.
- Save a reusable reference dataset for production-style checks.
- Schedule recurring monitoring runs with Prefect.
- Visualize drift metrics in Grafana from a Postgres table.
- Retrain and promote a refreshed model when drift signals justify it.

## Learning Path

| File / Folder | Description |
|---|---|
| [**01 - Train an ML Model**](01-train-ml-model.ipynb) | Train a baseline trip-duration model and register it in MLflow. |
| [**02 - Intro to Model and Data Drift**](02-intro-to-drift.ipynb) | Compare two months of taxi rides and measure drift with statistical distances. |
| [**03 - Intro to Evidently**](03-intro-to-evidently.ipynb) | Create Evidently reports and save a reference dataset for later monitoring. |
| [**04 - Evidently in Batch Predictions**](04-evidently-in-batch-predictions.md) | Schedule batch monitoring with Prefect and visualize metrics in Grafana. |
| [**05 - Retraining After Drift**](05-retrain-after-drift.ipynb) | Retrain the model on a fresh labeled batch and promote it if it improves the holdout score. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**src**](src/) | The Prefect batch monitoring flow and its deployment script used in chapter 04. |
| [**grafana**](grafana/) | Provisioned Postgres data source and drift dashboard for the local Grafana instance. |
| [**data**](data/) | Local storage for the downloaded taxi Parquet files and the saved reference dataset. |
| [**assets**](assets/) | Visual aids referenced in the repository. |
| [**docker-compose.yaml**](docker-compose.yaml) | The local Postgres and Grafana services used in chapter 04. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Local Data Services

The batch monitoring chapter uses a local monitoring stack defined in [docker-compose.yaml](docker-compose.yaml):

- **Postgres** stores drift metrics produced by the Prefect flow.
- **Grafana** reads that table and displays the provisioned monitoring dashboard.

The default local endpoints are:

- Grafana: `http://127.0.0.1:3000`
- Postgres: `localhost:5555`
- Prefect UI: `http://127.0.0.1:4200`

## Prerequisites

- **Docker Desktop**: required to run the local Postgres and Grafana services in chapter 04. Follow the [installation instructions](https://docs.docker.com/get-docker/) if you do not have it yet. Make sure it is **installed and running** before you start that chapter.

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd mle-monitoring-in-ml`.

### 1. Use This Repository as a Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone Your Copy

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Create the Environment File

Copy the template once:

```bash
cp .env.example .env
```

Keep the default local URLs unless you intentionally change the local stack ports.

> [!CAUTION]
> Never commit `.env`. It is git-ignored because environment files often hold secrets. Only `.env.example`, with placeholder or default values, belongs in the repository.

---

### 6. Open the Repository in VS Code

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open [01-train-ml-model.ipynb](01-train-ml-model.ipynb) and select the Python environment created by `uv sync` as the kernel.

## References & Further Reading

- [Data Drift in ML Production](https://www.evidentlyai.com/ml-in-production/data-drift): What data drift is, why it happens, and how teams detect it.
- [Evidently Docs](https://docs.evidentlyai.com/): Reports, metrics and tests for data and prediction drift.
- [Evidently Data Drift Preset](https://docs.evidentlyai.com/metrics/preset_data_drift): How Evidently chooses drift tests and thresholds per column.
- [MLOps Zoomcamp: Model Monitoring](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/05-monitoring): A public course module that builds a similar Evidently, Prefect and Grafana monitoring stack.

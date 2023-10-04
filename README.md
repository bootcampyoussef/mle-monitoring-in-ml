# Monitoring in ML

Monitoring is a crucial part of the ML lifecycle. It is important to monitor the model performance and data quality in production. This is because the model performance and data quality can degrade over time. This can be due to changes in the data distribution, changes in the user behavior, or changes in the business logic. In the first notebook you will train a model on the green taxi dataset. The second notebook will introduce you to drift and how to detect it with statistical tests. In the third notebook you will find an introduction to evidently and how this tool helps you to automate the detection of drift. And in the last notebook you will see how to use evidently to monitor a batch prediction job with prefect.

## Notebooks

1. [Model Training](./01-train-ml-model.ipynb)
2. [Model Drift](./02-intro-to-drift.ipynb)
3. [Evidently](./03-intro-to-evidently.ipynb)
4. [Evidently in Batch predictions](./04-evidently-in-batch-predictions.ipynb)

## Setup

### Environment

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

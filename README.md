# Toxicity in OLID-BR (ToChiquinho)

ToChiquinho is a toxicity detection system for Brazilian Portuguese texts based on the [OLID-BR](https://github.com/DougTrajano/olid-br/) dataset.

## Modeling features

- Early stopping policy (patience).
- Imbalanced dataset handling.
- Hyperparameter optimization (Bayesian optimization).

## Usage

ToChiquinho is available as a Docker image. To run it, you need to have Docker installed on your machine.

Then, you can run the following command:

```bash
docker run -p 5000:5000 dougtrajano/tochiquinho
```

### Defining the task

ToChiquinho is a toxicity detection system with multiple tasks. You can parameterize the task that will be served by the API by setting the `API_TASK` environment variable. The available tasks are:

- `all`: serves all tasks (default) - Warning: this task is not recommended for production environments.
- `route`: serves the route task. This task is recommended for production environments.
- `toxic_spans`: serves the toxic spans detection task.
- `toxicity_target_type`: serves the toxicity target type identification task.
- `toxicity_target`: serves the toxicity target classification task.
- `toxicity_type`: serves the toxicity type detection task.
- `toxicity`: serves the toxicity classification task.

See more in the [Environment variables](#environment-variables) section.

## Environment variables

The following environment variables are needed to run the Jupyter Notebooks.

| Variable | Description |
|----------|-------------|
| `MLFLOW_TRACKING_URI` | The URI of the MLflow tracking server. |
| `MLFLOW_TRACKING_USERNAME` | The username of the MLflow tracking server. |
| `MLFLOW_TRACKING_PASSWORD` | The password of the MLflow tracking server. |
| `HUGGINGFACE_HUB_USERNAME` | The username of the Hugging Face Hub. |
| `HUGGINGFACE_HUB_TOKEN` | The token of the Hugging Face Hub. |
| `AWS_PROFILE` | The AWS profile to be used. |
| `SAGEMAKER_EXECUTION_ROLE_ARN` | The ARN of the SageMaker execution role. |

> **Note**: Some variables can be passed to the SageMaker training jobs.

The following environment variables are needed to run the training jobs.

| Variable | Description |
|----------|-------------|
| `MLFLOW_TRACKING_URI` | The URI of the MLflow tracking server. |
| `MLFLOW_EXPERIMENT_NAME` | The name of the MLflow experiment. |
| `MLFLOW_TRACKING_USERNAME` | The username of the MLflow tracking server. |
| `MLFLOW_TRACKING_PASSWORD` | The password of the MLflow tracking server. |
| `MLFLOW_TAGS` | The tags to be added to the MLflow run. |
| `MLFLOW_FLATTEN_PARAMS` | Whether to flatten the parameters. |
| `MLFLOW_RUN_ID` | The ID of the MLflow run. If set, a child run will be created. |
| `HF_MLFLOW_LOG_ARTIFACTS` | Whether to log artifacts to MLflow. |
| `HUGGINGFACE_HUB_TOKEN` | The token of the Hugging Face Hub. |
| `SM_TRAINING_ENV` | The JSON string of the SageMaker training environment. |
| `TOKENIZERS_PARALLELISM` | Whether to use parallelism for tokenizers. |

## Changelog

See the [GitHub Releases](https://github.com/DougTrajano/ToChiquinho/releases) page for a history of notable changes to this project.

## Contributing

Contributions are welcome! Please open a pull request and include a detailed description of your changes.

## Development environment setup

To set up the development environment, you need to have the following tools installed on your machine:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Python 3.10](https://www.python.org/)

Then, you can run the following commands in the root directory to install the dependencies:

```bash
pip install -r requirements-dev.txt
pip install -r requirements-docs.txt
pip install -r requirements.txt
```

You also need to install the [pre-commit](https://pre-commit.com/) hooks:

```bash
pre-commit install
```

### Add possible secrets to the .secrets.baseline file

If you get an error when running the pre-commit hooks, you may need to add the possible secrets to the `.secrets.baseline` file. To do so, you can run the following command:

```bash
detect-secrets scan --baseline .secrets.baseline
```

For more information, see the [Yelp/detect-secrets](https://github.com/Yelp/detect-secrets).

### Running the tests

To run the tests, you can run the following command in the root directory:

```bash
pytest
```

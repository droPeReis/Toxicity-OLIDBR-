import os
import json
from pydantic import BaseSettings, Field
from typing import Optional, Dict, Any


class EnvironmentVariables(BaseSettings):
    MLFLOW_TRACKING_URI: Optional[str] = Field(
        None, description="MLflow tracking URI"
    )
    MLFLOW_EXPERIMENT_NAME: Optional[str] = Field(
        None, description="MLflow experiment name"
    )
    MLFLOW_TRACKING_USERNAME: Optional[str] = Field(
        None, description="MLflow tracking username"
    )
    MLFLOW_TRACKING_PASSWORD: Optional[str] = Field(
        None, description="MLflow tracking password"
    )
    MLFLOW_TAGS: Optional[Dict[str, str]] = Field(
        None, description="MLflow tags"
    )
    MLFLOW_FLATTEN_PARAMS: Optional[bool] = Field(
        None, description="MLflow flatten params"
    )
    MLFLOW_RUN_ID: Optional[str] = Field(None, description="MLflow run ID")
    HF_MLFLOW_LOG_ARTIFACTS: Optional[bool] = Field(
        None, description="HuggingFace MLflow log artifacts"
    )
    HUGGINGFACE_HUB_TOKEN: Optional[str] = Field(
        None, description="HuggingFace Hub token"
    )

    SM_TRAINING_ENV: Optional[Dict[str, Any]] = Field(
        None, description="Sagemaker training environment"
    )
    TOKENIZERS_PARALLELISM: Optional[bool] = Field(
        None, description="HuggingFace Tokenizers parallelism"
    )

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        os.environ[name] = str(value)

import os
import time
import json
import torch
import mlflow
import datasets
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Union, Dict, List, Tuple
from huggingface_hub.repository import Repository
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Custom code
from arguments import TrainScriptArguments
from environments import EnvironmentVariables
from logger import setup_logger
from metrics.utils import compute_metrics
from utils import flatten_dict

_logger = setup_logger(__name__)


class Experiment(object):
    name = "base-experiment"

    def __init__(self, args: TrainScriptArguments):
        """Initialize the experiment.

        Args:
        - args: The arguments.
        """
        self.args = args
        self.env = EnvironmentVariables()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.job_name = (
            self.env.SM_TRAINING_ENV.get("job_name")
            if self.env.SM_TRAINING_ENV
            else None
        )

        if self.job_name:
            self.args.output_dir = os.path.join(
                self.args.output_dir, self.job_name
            )

        self.model_output_dir = f"{self.args.output_dir}/model"
        self.prep_output_dir(self.args.output_dir)

        _logger.debug(
            {
                "args": self.args,
                "device": self.device,
                "job_name": self.job_name,
            }
        )

        _logger.info(f"Experiment {self.name} initialized.")

    def __str__(self):
        """String representation of the experiment."""
        return self.name

    def init_experiment(self):
        """Run the experiment."""
        _logger.info(f"Initializing experiment {self.name}.")

        set_seed(self.args.seed)

        if self.resume_mlflow_checkpoint(self.args.output_dir):
            _logger.info("Resuming MLflow from checkpoint.")
            self.env.MLFLOW_RUN_ID = self.resume_mlflow_checkpoint(
                self.args.output_dir
            )

        self.nested_run = bool(
            self.env.MLFLOW_RUN_ID
            and not self.resume_mlflow_checkpoint(self.args.output_dir)
        )
        _logger.debug(f"Nested run: {self.nested_run}")

        if self.env.MLFLOW_RUN_ID and self.nested_run:
            mlflow.start_run()
            _logger.debug(
                f"Starting mlflow run: {mlflow.active_run().info.run_id}"
            )

    def init_model(self, pretrained_model_name_or_path: str):
        """Initialize the model.

        Args:
        - pretrained_model_name_or_path: The pretrained model name or path.

        Returns:
        - The model.
        """
        _logger.info(
            f"Initializing model from {pretrained_model_name_or_path}."
        )
        self.model = PreTrainedModel.from_pretrained(
            pretrained_model_name_or_path
        ).to(self.device)
        mlflow.log_text(str(self.model), "model_summary.txt")
        return self.model

    def init_tokenizer(self, pretrained_model_name_or_path: str):
        """Initialize the tokenizer.

        Args:
        - pretrained_model_name_or_path: The pretrained model name or path.

        Returns:
        - The tokenizer.
        """
        _logger.info(
            f"Initializing tokenizer from {pretrained_model_name_or_path}."
        )

        # Disable parallelism to avoid OOM errors.
        self.env.TOKENIZERS_PARALLELISM = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )

        return self.tokenizer

    def load_dataset(self):
        """Load the dataset."""
        _logger.info(f"Loading dataset from {self.args.data_dir}.")
        dataset = datasets.load_from_disk(self.args.data_dir)
        return dataset

    def slice_dataset(
        self, dataset: Union[datasets.Dataset, datasets.DatasetDict]
    ):
        """Slice the dataset.

        Args:
        - dataset: The dataset.

        Returns:
        - The sliced dataset.
        """
        max_samples = {
            "train": self.args.max_train_samples,
            "validation": self.args.max_val_samples,
            "test": self.args.max_test_samples,
        }

        if not any(max_samples.values()):
            return dataset

        _logger.info("Slicing dataset.")
        for key, value in max_samples.items():
            if value is not None:
                dataset[key] = dataset[key].select(range(value))
                _logger.info(f"Sliced {key} dataset to {value} samples.")

        for split in dataset.keys():
            mlflow.log_param(f"{split}_size", len(dataset[split]))

        _logger.info(f"Dataset: {dataset}")
        return dataset

    def prepare_dataset(
        self, dataset: Union[datasets.Dataset, datasets.DatasetDict]
    ):
        """Prepare the dataset.

        Args:
        - dataset: The dataset.

        Returns:
        - The prepared dataset.
        """
        _logger.info("Preparing dataset.")

        # Concatenate the train and validation sets to create a new train set.
        if (
            self.args.concat_validation_set
            and self.args.eval_dataset != "validation"
            and "train" in dataset.keys()
            and "validation" in dataset.keys()
        ):
            _logger.info("Concatenating validation set to train set.")
            self.dataset["train"] = datasets.concatenate_datasets(
                [self.dataset["train"], self.dataset["validation"]]
            )

        _logger.info(f"Dataset: {dataset}")
        return dataset

    def get_dataset_stats(
        self, dataset: Union[datasets.Dataset, datasets.DatasetDict]
    ):
        """Get the dataset statistics (e.g. length, average qty words, etc.).

        Args:
        - dataset: The dataset.

        Returns:
        - The dataset statistics as a dictionary.
        """
        _logger.info("Getting dataset statistics.")

        def compute_stats(examples):
            return {
                "min_length": min([len(x) for x in examples["text"]]),
                "max_length": max([len(x) for x in examples["text"]]),
                "avg_length": round(
                    np.mean([len(x) for x in examples["text"]]), 2
                ),
                "min_words": min([len(x.split()) for x in examples["text"]]),
                "max_words": max([len(x.split()) for x in examples["text"]]),
                "avg_words": round(
                    np.mean([len(x.split()) for x in examples["text"]]), 2
                ),
            }

        dataset_stats = {
            split: compute_stats(dataset[split]) for split in dataset.keys()
        }

        _logger.info(f"Dataset statistics: {dataset_stats}")
        return dataset_stats

    def prep_output_dir(self, output_dir: str) -> bool:
        """Prepare the output directory.

        Args:
        - output_dir: The output directory.

        """
        # Create the output directory if it doesn't exist.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            _logger.info(f"Created output directory {output_dir}.")

        # Create the model subdirectory if it doesn't exist.
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)
            _logger.info(f"Created model directory {self.model_output_dir}.")

    def save_mlflow_checkpoint(
        self,
        mlflow_run_id: str,
        output_dir: str,
        file_name: str = "mlflow_run_id.txt",
    ) -> None:
        """Save the MLFLOW_RUN_ID to the checkpoint directory.

        Args:
        - mlflow_run_id: The MLFLOW_RUN_ID.
        - output_dir: The output directory.
        - file_name: The file name to save the MLFLOW_RUN_ID.
        """
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        path = os.path.join(output_dir, file_name)

        with open(path, "w") as f:
            f.write(mlflow_run_id)
            _logger.info(f"Saved MLFLOW_RUN_ID {mlflow_run_id} to {path}.")

    def resume_mlflow_checkpoint(
        self, output_dir: str, file_name: str = "mlflow_run_id.txt"
    ) -> Union[str, None]:
        """Read the MLFLOW_RUN_ID from the checkpoint directory.

        Args:
        - output_dir: The output directory.
        - file_name: The file name to save the MLFLOW_RUN_ID.

        Returns:
        - The MLFLOW_RUN_ID.
        """
        path = os.path.join(output_dir, file_name)
        if os.path.isfile(path):
            with open(path, "r") as f:
                mlflow_run_id = f.read()
                _logger.info(
                    f"Read MLFLOW_RUN_ID {mlflow_run_id} from {path}."
                )
                return mlflow_run_id

    def plot_hf_metrics(
        self,
        log_history: List[Dict[str, float]],
        metrics: Dict[str, str] = {
            "eval_accuracy": "Accuracy",
            "eval_f1": "F1-score",
            "eval_precision": "Precision",
            "eval_recall": "Recall",
        },
        xtitle: str = "Epoch",
        ytitle: str = "Scores",
        ylim: Union[Tuple[float, float], None] = (0.0, 1.0),
    ) -> plt.Figure:
        """Plot the Hugging Face metrics.

        Args:
        - log_history: The Hugging Face log history.
        - metrics: The metrics to plot (key: metric name, value: plot title).
        - xtitle: The x-axis title.
        - ytitle: The y-axis title.
        - ylim: The y-axis limits.

        Returns:
        - The plot.
        """
        _logger.debug(
            {
                "log_history": log_history,
                "metrics": metrics,
                "xtitle": xtitle,
                "ytitle": ytitle,
            }
        )

        # Prepare the metrics
        _metrics = OrderedDict()
        for item in log_history:
            epoch = int(item["epoch"])
            for key, value in item.items():
                if key in metrics:
                    if epoch not in _metrics:
                        _metrics[epoch] = {}
                    _metrics[epoch][key] = value

        fig = plt.figure(figsize=(10, 6))
        for key, value in metrics.items():
            data = [_metrics[i][key] for i in _metrics if key in _metrics[i]]
            if len(data) == 0:
                raise ValueError(
                    f"{key} is not in the metrics. Metrics: {_metrics}."
                )
            plt.plot(data, label=value)

        plt.xticks(range(len(_metrics)), range(1, len(_metrics) + 1))

        if ylim is not None:
            plt.ylim(ylim)

        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.legend()

        return fig

    def add_sm_patterns_to_gitignore(
        self,
        repo: Repository,
        gitignore_path: str = ".gitignore",
        patterns: List[str] = [
            "*.sagemaker-uploading",
            "*.sagemaker-uploaded",
        ],
    ) -> None:
        """Add the SageMaker Checkpointing patterns to the .gitignore file in Model Repository.

        Args:
        - repo: The Model Repository.
        """
        _logger.debug(
            f"Adding SageMaker Checkpointing patterns to {repo.local_dir}/{gitignore_path}."
        )

        # Check if .gitignore exists
        if os.path.exists(os.path.join(repo.local_dir, gitignore_path)):
            with open(os.path.join(repo.local_dir, gitignore_path), "r") as f:
                current_content = f.read()
        else:
            current_content = ""

        # Add the patterns to .gitignore
        content = current_content
        for pattern in patterns:
            if pattern not in content:
                _logger.debug(
                    f"Adding {pattern} to {repo.local_dir}/{gitignore_path}."
                )
                if content.endswith("\n"):
                    content += pattern
                else:
                    content += f"\n{pattern}"

        with open(os.path.join(repo.local_dir, gitignore_path), "w") as f:
            _logger.debug(f"Writing .gitignore file. Content: {content}")
            f.write(content)

        repo.git_add(gitignore_path)

        # avoid race condition with git status
        time.sleep(1)

        if not repo.is_repo_clean():
            repo.git_commit(f"Add *.sagemaker patterns to {gitignore_path}.")
            repo.git_push()

    def run(self):
        """Run the training."""
        self.init_experiment()
        with mlflow.start_run(nested=self.nested_run):
            # Save MLflow run ID to checkpointing directory.
            self.save_mlflow_checkpoint(
                mlflow_run_id=mlflow.active_run().info.run_id,
                output_dir=self.args.output_dir,
            )

            self.init_tokenizer(self.args.model_name)

            self.dataset = self.load_dataset()
            self.dataset = self.slice_dataset(self.dataset)
            self.dataset = self.prepare_dataset(self.dataset)

            # It should be done before the tokenization.
            dataset_stats = self.get_dataset_stats(self.dataset)
            mlflow.log_params(flatten_dict(dataset_stats))

            self.init_model(self.args.model_name)

            self.dataset.set_format("torch")

            trainer_args = TrainingArguments(
                output_dir=self.args.output_dir,
                overwrite_output_dir=True
                if get_last_checkpoint(self.model_output_dir) is not None
                else False,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=self.args.early_stopping_patience + 1,
                load_best_model_at_end=True,
                push_to_hub=self.args.push_to_hub,
                hub_token=self.env.HUGGINGFACE_HUB_TOKEN,
                hub_model_id=self.args.hub_model_id,
                metric_for_best_model="f1",
                learning_rate=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                adam_beta1=self.args.adam_beta1,
                adam_beta2=self.args.adam_beta2,
                adam_epsilon=self.args.adam_epsilon,
                label_smoothing_factor=self.args.label_smoothing_factor,
                optim=self.args.optim,
                per_device_train_batch_size=self.args.batch_size,
                per_device_eval_batch_size=self.args.batch_size,
                num_train_epochs=self.args.num_train_epochs,
                report_to=["mlflow"],
                seed=self.args.seed,
            )

            trainer = Trainer(
                model=self.model,
                args=trainer_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset[self.args.eval_dataset],
                tokenizer=self.tokenizer,
                compute_metrics=lambda p: compute_metrics(
                    p,
                    threshold=self.args.threshold,
                    problem_type="multi-class",
                ),
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.args.early_stopping_patience
                    )
                ]
                if self.args.early_stopping_patience is not None
                else None,
            )

            # Add *.sagemaker-uploading and *.sagemaker-uploaded patterns to .gitignore.
            self.add_sm_patterns_to_gitignore(trainer.repo)

            # check if checkpoint existing if so continue training
            last_checkpoint = get_last_checkpoint(self.model_output_dir)
            if last_checkpoint is not None:
                _logger.info(
                    f"Resuming training from checkpoint: {last_checkpoint}"
                )

            trainer.train(resume_from_checkpoint=last_checkpoint)

            # Evaluate model
            mlflow.log_param("eval_dataset", self.args.eval_dataset)
            scores = trainer.evaluate()
            _logger.info(f"Scores ({self.args.eval_dataset} set): {scores}")
            _logger.info(f"eval_f1_weighted: {scores['eval_f1']}")

            # Plot metrics
            _logger.info("Plotting scores.")
            mlflow.log_figure(
                figure=self.plot_hf_metrics(
                    log_history=trainer.state.log_history
                ),
                artifact_file="scores.png",
            )

            # Plot loss
            _logger.info("Plotting losses.")
            mlflow.log_figure(
                figure=self.plot_hf_metrics(
                    log_history=trainer.state.log_history,
                    metrics={"eval_loss": "Loss"},
                    xtitle="Epoch",
                    ytitle="Loss",
                    ylim=None,
                ),
                artifact_file="losses.png",
            )

            mlflow.log_dict(
                dictionary=trainer.state.log_history,
                artifact_file="log_history.json",
            )

            trainer.save_model(self.args.model_dir)

            if self.args.push_to_hub:
                _logger.info("Pushing model to Hugging Face Hub.")
                trainer.push_to_hub(
                    blocking=True,
                    language="pt",
                    license="apache-2.0",
                    tags=[
                        "toxicity",
                        "portuguese",
                        "hate speech",
                        "offensive language",
                    ],
                    model_name=self.args.hub_model_id,
                    finetuned_from=self.args.model_name,
                    tasks="text-classification",
                    dataset="OLID-BR",
                )
                _logger.info("Model pushed to Hugging Face Hub.")

        _logger.info("Experiment completed.")

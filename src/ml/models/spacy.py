import os
import json
import random
import numpy as np
from typing import List, Union, Set

import spacy
import string
import itertools
import warnings
from pathlib import Path
from spacy.tokens import Doc
from spacy.training.example import Example
from sklearn.base import BaseEstimator
import seaborn as sns
import matplotlib.pyplot as plt
from thinc.optimizers import Optimizer

from metrics.spans import f1_score
from logger import setup_logger

_logger = setup_logger(__name__)


def _contiguous_ranges(span_list: List[int]):
    """Extracts continguous runs [1, 2, 3, 5, 6, 7] -> [(1,3), (5,7)].

    Args:
    - span_list: a list of span indicies

    Returns:
    - A list of tuples containing the start and end of each continguous run.
    """
    output = []
    for _, span in itertools.groupby(
        enumerate(span_list), lambda p: p[1] - p[0]
    ):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    return output


def fix_spans(
    spans: List[int], text: str, special_characters: str = string.whitespace
):
    """Applies minor edits to trim spans and remove singletons.

    Args:
    - spans: a list of span indicies
    - text: the text to which the spans apply
    - special_characters: a string containing special characters to remove from the text

    Returns:
    - A list of fixed spans.
    """
    cleaned = []
    for begin, end in _contiguous_ranges(spans):
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        if end - begin > 1:
            cleaned.extend(range(begin, end + 1))
    return cleaned


def spans_to_ents(doc: Doc, spans: Set[int], label: str):
    """Converts span indicies into spacy entity labels.

    Args:
    - doc: a spacy Doc object
    - spans: a list of span indicies
    - label: the entity label to assign to the spans

    Returns:
    - A list containing start, end, and label.
    """
    started = False
    left, right, ents = 0, 0, []
    for x in doc:
        if x.pos_ == "SPACE":
            continue
        if spans.intersection(set(range(x.idx, x.idx + len(x.text)))):
            if not started:
                left, started = x.idx, True
            right = x.idx + len(x.text)
        elif started:
            ents.append((left, right, label))
            started = False
    if started:
        ents.append((left, right, label))
    return ents


class ToxicSpansDetectionModel(BaseEstimator):
    def __init__(
        self, spacy_model: str = "pt_core_news_lg", toxic_label: str = "TOXIC"
    ):
        """Initializes the model.

        Args:
        - spacy_model: the spaCy model to use.
        - toxic_label: the label to use for toxic spans.
        """
        self.nlp = spacy.load(spacy_model)
        self.toxic_label = toxic_label
        self._model = None
        self._trained_epochs = None

        self.losses = []
        self.train_scores = []
        self.eval_scores = []
        self.saved_models = {}

        sns.set_theme(
            style="white",
            rc={"axes.spines.right": False, "axes.spines.top": False},
        )

        _logger.info("Initialized ToxicSpansDetection model.")

    def init_model(self) -> spacy.language.Language:
        """Initializes the model.

        Returns:
        - The initialized model.
        """
        _logger.debug("Initializing ToxicSpansDetection model.")
        model = spacy.blank("pt")
        model.vocab.strings.add(self.toxic_label)
        ner = self.nlp.create_pipe("ner")
        model.add_pipe("ner", last=True)
        ner.add_label(self.toxic_label)
        _logger.debug("ToxicSpansDetection model initialized.")
        return model

    def init_optimizer(
        self,
        optim: str = "sgd",
        learning_rate: float = 0.001,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> Optimizer:
        """Initializes the optimizer.

        Args:
        - optim: the optimizer to use (adam, radam or sgd).
        - learning_rate: the learning rate to use.
        - adam_beta1: the beta1 parameter to use for the Adam optimizer.
        - adam_beta2: the beta2 parameter to use for the Adam optimizer.
        - adam_epsilon: the epsilon parameter to use for the Adam optimizer.
        - weight_decay: the weight decay to use.

        Returns:
        - The thinc optimizer.
        """
        if optim == "adam":
            _logger.debug("Initializing Adam optimizer.")
            from thinc.optimizers import Adam

            optimizer = Adam(
                learn_rate=learning_rate,
                beta1=adam_beta1,
                beta2=adam_beta2,
                eps=adam_epsilon,
                L2=weight_decay,
            )
        elif optim == "radam":
            _logger.debug("Initializing RAdam optimizer.")
            from thinc.optimizers import RAdam

            optimizer = RAdam(
                learn_rate=learning_rate,
                beta1=adam_beta1,
                beta2=adam_beta2,
                eps=adam_epsilon,
                L2=weight_decay,
            )
        elif optim == "sgd":
            _logger.debug("Initializing SGD optimizer.")
            from thinc.optimizers import SGD

            optimizer = SGD(learn_rate=learning_rate, L2=weight_decay)
        else:
            _logger.warning(f"Optimizer {optim} not supported. Using SGD.")
            _logger.debug("Initializing SGD optimizer.")
            from thinc.optimizers import SGD

            optimizer = SGD(learn_rate=0.001, L2=weight_decay)
        _logger.debug("Optimizer initialized.")
        return optimizer

    def load_model_from_checkpoint(
        self, checkpoint_dir: str, metrics_dir: str = None
    ):
        """Loads the model from the checkpoint directory.

        Args:
        - checkpoint_dir: the directory to load the model from.
        - metrics_dir: the directory to load the metrics from.

        Path: {checkpoint_dir}/model_{epoch}/...

        Returns:
        - The loaded model.
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            _logger.debug("Checkpoint directory does not exist.")
            return None

        _logger.debug(f"Loading model from checkpoint: {checkpoint_dir}")

        if self._model is None:
            self._model = self.init_model()
        self._model = self._model.from_disk(checkpoint_dir)

        _logger.debug("Model loaded.")
        self._trained_epochs = int(checkpoint_dir.__str__().split("_")[-1])

        # Load metrics
        _logger.debug("Loading metrics from checkpoint.")
        if metrics_dir:
            metrics_dir = Path(metrics_dir)
        else:
            metrics_dir = checkpoint_dir

        with open(metrics_dir / "metrics.json", "r") as f:
            metrics = json.load(f)
            self.losses = metrics["losses"]
            self.train_scores = metrics["train_scores"]
            self.eval_scores = metrics["eval_scores"]

        return self._model

    def save(self, checkpoint_dir: Union[str, Path], overwrite: bool = False):
        """Saves the model to the checkpoint directory.

        Args:
        - checkpoint_dir: the directory to save the model to.
        - overwrite: whether to overwrite the existing model.
        """
        _logger.debug("Saving ToxicSpansDetection model.")

        checkpoint_dir = Path(checkpoint_dir)

        if checkpoint_dir.exists() and not overwrite:
            _logger.error("Checkpoint directory already exists.")
            raise FileExistsError(
                f"Checkpoint directory already exists: {checkpoint_dir}."
            )

        self._model.to_disk(checkpoint_dir)

        # Save metrics
        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "losses": self.losses,
                    "train_scores": self.train_scores,
                    "eval_scores": self.eval_scores,
                },
                f,
            )

        _logger.debug("ToxicSpansDetection model saved.")

    def get_latest_checkpoint(self, checkpoint_dir: str):
        """Finds the latest checkpoint in the checkpoint directory.

        Args:
        - checkpoint_dir: the directory to search for checkpoints in.

        Returns:
        - The path to the latest checkpoint.
        """
        _logger.debug("Finding latest ToxicSpansDetection model checkpoint.")

        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            _logger.debug("Checkpoint directory does not exist.")
            return None

        # Find the latest checkpoint
        epochs = []
        for checkpoint in checkpoint_dir.glob("model_*"):
            if checkpoint.is_dir():
                epochs.append(int(checkpoint.name.split("_")[-1]))
        if len(epochs) == 0:
            _logger.debug("No checkpoints found.")
            return None
        latest_checkpoint = checkpoint_dir / f"model_{max(epochs)}"
        _logger.debug(f"Latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint

    def fit(
        self,
        x: List[str],
        y: List[int],
        x_val: List[str] = None,
        y_val: List[int] = None,
        eval_every: int = 1,
        early_stopping_patience: int = None,
        epochs: int = 30,
        dropout: float = 0.1,
        optim: str = "sgd",
        learning_rate: float = 0.001,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        checkpoint_dir: str = "checkpoints",
        load_best_model_at_end: bool = True,
    ):
        """Fits the model.

        Args:
        - X: the list of texts.
        - y: the list of labels.
        - X_val: the list of validation texts.
        - y_val: the list of validation labels.
        - eval_every: the number of epochs between evaluations.
        - early_stopping_patience: the number of epochs to wait before early stopping.
        - epochs: the number of epochs to train the model.
        - dropout: the dropout rate to use.
        - optim: the optimizer to use (adam, radam or sgd).
        - learning_rate: the learning rate to use.
        - adam_beta1: the beta1 parameter to use for the Adam optimizer.
        - adam_beta2: the beta2 parameter to use for the Adam optimizer.
        - adam_epsilon: the epsilon parameter to use for the Adam optimizer.
        - weight_decay: the weight decay to use.
        - checkpoint_dir: the directory to save checkpoints.
        - load_best_model_at_end: whether to load the best model at the end of training.
        """
        _logger.info("Training ToxicSpansDetection model.")

        if x_val is None or y_val is None:
            warnings.warn(
                "No validation data provided. "
                "Early stopping will not be used. "
                "The model will be trained on all epochs. "
            )

        training_data = []
        for _, (text, spans) in enumerate(zip(x, y)):
            doc = self.nlp(text)
            ents = spans_to_ents(doc, set(spans), self.toxic_label)
            training_data.append((doc.text, {"entities": ents}))

        self._model = self.init_model()

        # Load latest checkpoint
        if self.get_latest_checkpoint(checkpoint_dir):
            latest_checkpoint_dir = self.get_latest_checkpoint(checkpoint_dir)
            self._model = self.load_model_from_checkpoint(
                latest_checkpoint_dir
            )

        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        unaffected_pipes = [
            pipe
            for pipe in self._model.pipe_names
            if pipe not in pipe_exceptions
        ]

        with self._model.disable_pipes(*unaffected_pipes):
            optimizer = self.init_optimizer(
                optim=optim.lower(),
                learning_rate=learning_rate,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                adam_epsilon=adam_epsilon,
                weight_decay=weight_decay,
            )

            if self.get_latest_checkpoint(checkpoint_dir):
                optimizer = self._model.resume_training(sgd=optimizer)
            else:
                optimizer = self._model.begin_training(sgd=optimizer)

            self._trained_epochs = 0
            for epoch in range(epochs):
                _logs = {}
                losses = {}

                random.shuffle(training_data)
                batches = spacy.util.minibatch(
                    items=training_data,
                    size=spacy.util.compounding(4.0, 64.0, 1.01),
                )

                if self._trained_epochs and epoch < self._trained_epochs:
                    _logger.debug(f"Skipping epoch {epoch+1}.")
                    continue
                _logger.debug(f"Training epoch {epoch+1}.")

                for batch in batches:
                    examples = []
                    for text, annotations in batch:
                        example = Example.from_dict(
                            self.nlp.make_doc(text), annotations
                        )
                        examples.append(example)
                    self._model.update(
                        examples, drop=dropout, losses=losses, sgd=optimizer
                    )
                self.losses.append(losses["ner"])
                _logs["loss"] = f"{losses['ner']:.2f}"

                if epoch % eval_every == 0:
                    self.saved_models[epoch] = self._model
                    self.train_scores.append(self.score(x, y))
                    _logs["train_f1"] = f"{self.train_scores[-1]:.4f}"
                    if x_val and y_val:
                        self.eval_scores.append(self.score(x_val, y_val))
                        _logs["eval_f1"] = f"{self.eval_scores[-1]:.4f}"
                    self.save(f"{checkpoint_dir}/model_{epoch+1}")

                _logger.info(f"Epoch {epoch+1}/{epochs}. {_logs}.")

                self._trained_epochs += 1

                if (
                    x_val
                    and y_val
                    and early_stopping_patience
                    and self.early_stopping(
                        scores=self.eval_scores,
                        patience=early_stopping_patience,
                    )
                ):
                    _logger.info(f"Early stopping at epoch {epoch+1}.")
                    break

        if load_best_model_at_end and len(self.eval_scores) > 1:
            _logger.debug("Loading best model at end of training.")
            self.best_epoch = np.argmax(self.eval_scores) + 1
            self._model = self.load_model_from_checkpoint(
                checkpoint_dir=f"{checkpoint_dir}/model_{self.best_epoch}"
            )

            _logger.info(
                f"Best model found at epoch {self.best_epoch} with "
                f"F1-score (validation set) {self.eval_scores[self.best_epoch-1]:.4f}."
            )

    def early_stopping(self, scores: List[float], patience: int = 1) -> bool:
        """Checks if early stopping should be used.

        Args:
        - scores: the list of scores.
        - patience: the number of epochs to wait before early stopping.

        Returns:
        - whether to use early stopping.
        """
        if len(scores) < 2:
            return False

        best_score = scores[0]
        best_score_idx = 0
        for i in range(1, len(scores)):
            if scores[i] > best_score:
                best_score = scores[i]
                best_score_idx = i

        if (len(scores) - best_score_idx) > patience:
            return True
        else:
            return False

    def _predict(self, text: str) -> List[int]:
        """Predicts the toxic spans for a given text.

        Args:
        - text: the text.

        Returns:
        - the toxic spans as a list of integers.
        """
        if not self._model:
            raise ValueError("Model not trained or loaded yet.")

        preds = []
        doc = self._model(text)
        for ent in doc.ents:
            preds.extend(range(ent.start_char, ent.end_char))
        return preds

    def predict(
        self, x: Union[List[str], str]
    ) -> Union[List[int], List[List[int]]]:
        """Predicts the labels.

        Args:
        - x: the list of texts or a single text.

        Returns:
        - the toxic spans as a list of integers or a list of lists of integers.
        """
        if isinstance(x, (list, tuple, np.ndarray)):
            preds = []
            for text in x:
                preds.append(self._predict(text))
            return preds
        elif isinstance(x, str):
            return self._predict(x)
        else:
            raise ValueError("X must be a list of strings or a single string.")

    def score(self, x: List[str], y: List[int]):
        """Scores the model using the F1-score.

        Args:
        - x: the list of texts.
        - y: the list of labels.

        Returns:
        - the F1 score.
        """
        _logger.debug("Scoring model.")
        y_pred = self.predict(x)
        score = f1_score(y, y_pred)
        _logger.debug(f"F1-score: {score:.4f}")
        return score

    def plot_losses(
        self, xlabel: str = "Epoch", ylabel: str = "Loss"
    ) -> plt.Figure:
        """Plots the losses.

        Args:
        - xlabel: the x-axis label.
        - ylabel: the y-axis label.
        """
        _logger.debug("Plotting losses.")
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.losses, figure=fig)
        plt.xticks(
            ticks=[int(i) for i in range(len(self.losses))],
            labels=[int(i + 1) for i in range(len(self.losses))],
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return fig

    def plot_scores(
        self, xlabel: str = "Step", ylabel: str = "F1-score"
    ) -> plt.Figure:
        """Plots the scores.

        Args:
        - xlabel: the x-axis label.
        - ylabel: the y-axis label.
        """
        _logger.debug("Plotting scores.")
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.train_scores, label=f"{ylabel} (train)")
        plt.plot(self.eval_scores, label=f"{ylabel} (validation)")
        plt.xticks(
            ticks=[int(i) for i in range(len(self.train_scores))],
            labels=[int(i + 1) for i in range(len(self.train_scores))],
        )

        plt.ylim(0, 1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        return fig

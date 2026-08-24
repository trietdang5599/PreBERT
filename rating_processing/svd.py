"""Train-only matrix factorization with validation-based early stopping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


class SVD:
    """Biased matrix factorization used to produce user/item embeddings.

    Despite the historical class name, this model is optimized with gradient
    descent rather than a closed-form singular-value decomposition. Ratings are
    modeled as ``global_mean + user_bias + item_bias + user_vector·item_vector``.
    """

    MODEL_VERSION = 2

    def __init__(self, data_path: str | Path, num_factors: int, seed: int = 42):
        self.df = pd.read_csv(data_path)
        self.df["reviewerID"] = self.df["reviewerID"].astype(str)
        self.df["itemID"] = self.df["itemID"].astype(str)
        self.users = sorted(self.df["reviewerID"].unique())
        self.items = sorted(self.df["itemID"].unique())
        self.users_id_dict = {value: index for index, value in enumerate(self.users)}
        self.items_id_dict = {value: index for index, value in enumerate(self.items)}

        self.rows = self.df["reviewerID"].map(self.users_id_dict).to_numpy()
        self.cols = self.df["itemID"].map(self.items_id_dict).to_numpy()
        self.data = self.df["overall"].to_numpy(dtype=np.float64)
        self.k = num_factors
        self.learning_rate = 0.02
        # Sparse Amazon splits contain many one-interaction entities. Stronger
        # factor regularization prevents the latent vectors from memorizing
        # individual train ratings while the bias terms learn the main signal.
        self.factor_regularization = 0.05
        self.bias_regularization = 0.1
        self.iterations = 1000
        self.seed = seed
        self.global_mean = float(np.mean(self.data))

    def _create_embeddings(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 0.1 / np.sqrt(self.k), size=(count, self.k))

    def _observed_predictions(
        self,
        emb_user: np.ndarray,
        emb_item: np.ndarray,
        user_bias: np.ndarray,
        item_bias: np.ndarray,
    ) -> np.ndarray:
        return (
            self.global_mean
            + user_bias[self.rows]
            + item_bias[self.cols]
            + np.sum(emb_user[self.rows] * emb_item[self.cols], axis=1)
        )

    def _train_rmse(
        self,
        emb_user: np.ndarray,
        emb_item: np.ndarray,
        user_bias: np.ndarray,
        item_bias: np.ndarray,
    ) -> float:
        predictions = self._observed_predictions(
            emb_user, emb_item, user_bias, item_bias
        )
        return float(np.sqrt(np.mean((self.data - predictions) ** 2)))

    def _validation_rmse(
        self,
        frame: pd.DataFrame,
        emb_user: np.ndarray,
        emb_item: np.ndarray,
        user_bias: np.ndarray,
        item_bias: np.ndarray,
    ) -> float:
        predictions = np.full(len(frame), self.global_mean, dtype=np.float64)
        for position, row in enumerate(frame.itertuples(index=False)):
            user_index = self.users_id_dict.get(str(row.reviewerID))
            item_index = self.items_id_dict.get(str(row.itemID))
            if user_index is not None:
                predictions[position] += user_bias[user_index]
            if item_index is not None:
                predictions[position] += item_bias[item_index]
            if user_index is not None and item_index is not None:
                predictions[position] += np.dot(
                    emb_user[user_index], emb_item[item_index]
                )
        ratings = frame["overall"].to_numpy(dtype=np.float64)
        return float(np.sqrt(np.mean((ratings - predictions) ** 2)))

    def train(
        self,
        validation_data: pd.DataFrame | None = None,
        *,
        eval_every: int = 10,
        patience: int = 10,
    ) -> None:
        rng = np.random.default_rng(self.seed)
        emb_user = self._create_embeddings(len(self.users), rng)
        emb_item = self._create_embeddings(len(self.items), rng)
        user_bias = np.zeros(len(self.users), dtype=np.float64)
        item_bias = np.zeros(len(self.items), dtype=np.float64)
        user_counts = np.maximum(
            np.bincount(self.rows, minlength=len(self.users)), 1
        )
        item_counts = np.maximum(
            np.bincount(self.cols, minlength=len(self.items)), 1
        )

        best_metric = np.inf
        best_parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        stale_evaluations = 0

        for iteration in range(1, self.iterations + 1):
            predictions = self._observed_predictions(
                emb_user, emb_item, user_bias, item_bias
            )
            errors = predictions - self.data
            grad_user = np.zeros_like(emb_user)
            grad_item = np.zeros_like(emb_item)
            grad_user_bias = np.zeros_like(user_bias)
            grad_item_bias = np.zeros_like(item_bias)
            np.add.at(grad_user, self.rows, errors[:, None] * emb_item[self.cols])
            np.add.at(grad_item, self.cols, errors[:, None] * emb_user[self.rows])
            np.add.at(grad_user_bias, self.rows, errors)
            np.add.at(grad_item_bias, self.cols, errors)
            grad_user = (
                grad_user / user_counts[:, None]
                + self.factor_regularization * emb_user
            )
            grad_item = (
                grad_item / item_counts[:, None]
                + self.factor_regularization * emb_item
            )
            grad_user_bias = (
                grad_user_bias / user_counts + self.bias_regularization * user_bias
            )
            grad_item_bias = (
                grad_item_bias / item_counts + self.bias_regularization * item_bias
            )
            np.clip(grad_user, -5.0, 5.0, out=grad_user)
            np.clip(grad_item, -5.0, 5.0, out=grad_item)
            np.clip(grad_user_bias, -5.0, 5.0, out=grad_user_bias)
            np.clip(grad_item_bias, -5.0, 5.0, out=grad_item_bias)
            emb_user -= self.learning_rate * grad_user
            emb_item -= self.learning_rate * grad_item
            user_bias -= self.learning_rate * grad_user_bias
            item_bias -= self.learning_rate * grad_item_bias

            if iteration % eval_every != 0:
                continue
            train_rmse = self._train_rmse(
                emb_user, emb_item, user_bias, item_bias
            )
            metric = train_rmse
            message = f"SVD iteration {iteration}: train RMSE={train_rmse:.6f}"
            if validation_data is not None and len(validation_data):
                metric = self._validation_rmse(
                    validation_data, emb_user, emb_item, user_bias, item_bias
                )
                message += f", valid RMSE={metric:.6f}"
            print(message)

            if metric < best_metric - 1e-6:
                best_metric = metric
                best_parameters = (
                    emb_user.copy(),
                    emb_item.copy(),
                    user_bias.copy(),
                    item_bias.copy(),
                )
                stale_evaluations = 0
            else:
                stale_evaluations += 1
                if validation_data is not None and stale_evaluations >= patience:
                    print(f"SVD early stopped at iteration {iteration}")
                    break

        if best_parameters is None:
            best_parameters = (emb_user, emb_item, user_bias, item_bias)
        self.emb_user, self.emb_item, self.user_bias, self.item_bias = best_parameters
        self.best_metric = best_metric
        self.model_version = self.MODEL_VERSION

    def get_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "emb_user"):
            raise RuntimeError("SVD must be trained before requesting embeddings")
        return self.emb_user, self.emb_item

    def get_user_embedding(self, user_id: Any) -> np.ndarray:
        index = self.users_id_dict.get(str(user_id))
        if index is None:
            return np.mean(self.emb_user, axis=0)
        return self.emb_user[index]

    def get_item_embedding(self, item_id: Any) -> np.ndarray:
        index = self.items_id_dict.get(str(item_id))
        if index is None:
            return np.mean(self.emb_item, axis=0)
        return self.emb_item[index]


def initialize_svd(
    data_path: str | Path,
    num_factors: int,
    checkpoint_path: str | Path = "chkpt/svd_train.pt",
    validation_data: pd.DataFrame | None = None,
) -> SVD:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        svd = torch.load(checkpoint_path, weights_only=False)
        if getattr(svd, "model_version", 1) != SVD.MODEL_VERSION:
            checkpoint_path.unlink()
        else:
            if svd.k != num_factors:
                raise ValueError(
                    f"SVD checkpoint has {svd.k} factors, expected {num_factors}"
                )
            current = pd.read_csv(data_path)
            current_users = set(current["reviewerID"].astype(str))
            current_items = set(current["itemID"].astype(str))
            if current_users != set(svd.users) or current_items != set(svd.items):
                raise ValueError("SVD checkpoint was fitted on a different train split")
            return svd

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    svd = SVD(data_path, num_factors)
    svd.train(validation_data=validation_data)
    torch.save(svd, checkpoint_path)
    return svd

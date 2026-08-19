"""Factorization Machine fitted on train data and selected on validation RMSE."""

from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import OneHotEncoder


class Args:
    epochs = 100
    learning_rate = 0.02
    reg = 0.002
    batch_size = 1024
    patience = 10
    seed = 42


args = Args()


class FactorizationMachine:
    def __init__(
        self,
        n_factors: int,
        n_features: int,
        feature_names: np.ndarray,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_factors = n_factors
        self.w0 = 0.0
        self.w = np.zeros(n_features, dtype=np.float64)
        self.V = rng.normal(scale=0.01, size=(n_features, n_factors))
        self.feature_names = np.asarray(feature_names)

    def predict(self, X: csr_matrix) -> np.ndarray:
        X = X.tocsr()
        linear_terms = np.asarray(X @ self.w).ravel() + self.w0
        projected = np.asarray(X @ self.V)
        squared_projected = projected**2
        projected_squared = np.asarray(X.power(2) @ (self.V**2))
        interactions = 0.5 * np.sum(
            squared_projected - projected_squared,
            axis=1,
        )
        return linear_terms + interactions

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def fit(
        self,
        X: csr_matrix,
        y: np.ndarray,
        *,
        X_valid: csr_matrix | None,
        y_valid: np.ndarray | None,
        epochs: int,
        learning_rate: float,
        reg: float,
        batch_size: int,
        patience: int,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        best_metric = np.inf
        best_parameters: tuple[float, np.ndarray, np.ndarray] | None = None
        stale_epochs = 0

        for epoch in range(1, epochs + 1):
            order = rng.permutation(X.shape[0])
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                X_batch = X[indices].tocsr()
                y_batch = y[indices]
                predictions = self.predict(X_batch)
                errors = predictions - y_batch
                count = max(len(indices), 1)

                projected = np.asarray(X_batch @ self.V)
                grad_w0 = float(errors.mean())
                grad_w = np.asarray(X_batch.T @ errors).ravel() / count + reg * self.w
                grad_V = np.empty_like(self.V)
                X_squared = X_batch.power(2)
                squared_error_projection = np.asarray(
                    X_squared.T @ errors
                ).ravel()
                for factor in range(self.n_factors):
                    first = np.asarray(
                        X_batch.T @ (errors * projected[:, factor])
                    ).ravel()
                    second = self.V[:, factor] * squared_error_projection
                    grad_V[:, factor] = (
                        first - second
                    ) / count + reg * self.V[:, factor]

                np.clip(grad_w, -5.0, 5.0, out=grad_w)
                np.clip(grad_V, -5.0, 5.0, out=grad_V)
                self.w0 -= learning_rate * np.clip(grad_w0, -5.0, 5.0)
                self.w -= learning_rate * grad_w
                self.V -= learning_rate * grad_V

            train_rmse = self._rmse(y, self.predict(X))
            metric = train_rmse
            message = f"FM epoch {epoch}: train RMSE={train_rmse:.6f}"
            if X_valid is not None and y_valid is not None and len(y_valid):
                metric = self._rmse(y_valid, self.predict(X_valid))
                message += f", valid RMSE={metric:.6f}"
            print(message)

            if metric < best_metric - 1e-6:
                best_metric = metric
                best_parameters = (self.w0, self.w.copy(), self.V.copy())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if X_valid is not None and stale_epochs >= patience:
                    print(f"FM early stopped at epoch {epoch}")
                    break

        if best_parameters is not None:
            self.w0, self.w, self.V = best_parameters
        self.best_metric = best_metric

    def get_embedding(self, feature_name: str) -> np.ndarray:
        indices = np.where(self.feature_names == feature_name)[0]
        if len(indices):
            return self.V[int(indices[0])]
        prefix = feature_name.split("_", 1)[0] + "_"
        matching = np.char.startswith(self.feature_names.astype(str), prefix)
        if np.any(matching):
            return np.mean(self.V[matching], axis=0)
        return np.mean(self.V, axis=0)

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("wb") as handle:
            pickle.dump(
                {
                    "n_factors": self.n_factors,
                    "w0": self.w0,
                    "w": self.w,
                    "V": self.V,
                    "feature_names": self.feature_names,
                    "best_metric": getattr(self, "best_metric", None),
                },
                handle,
            )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        expected_feature_names = self.feature_names.copy()
        with Path(checkpoint_path).open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint["n_factors"] != self.n_factors:
            raise ValueError(
                "FM checkpoint factor count does not match the current experiment"
            )
        if not np.array_equal(checkpoint["feature_names"], expected_feature_names):
            raise ValueError("FM checkpoint was fitted on a different train split")
        self.w0 = checkpoint["w0"]
        self.w = checkpoint["w"]
        self.V = checkpoint["V"]
        self.feature_names = checkpoint["feature_names"]
        self.best_metric = checkpoint.get("best_metric")


def _interaction_frame(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    frame = pd.read_csv(data) if isinstance(data, (str, Path)) else data.copy()
    frame = frame[["overall", "reviewerID", "itemID"]]
    frame["reviewerID"] = frame["reviewerID"].astype(str)
    frame["itemID"] = frame["itemID"].astype(str)
    return frame


def run(
    file_path: str | Path,
    n_factors: int,
    checkpoint_path: str | Path,
    sparse_matrix_path: str | Path,
    validation_data: pd.DataFrame | None = None,
) -> FactorizationMachine:
    train_frame = _interaction_frame(file_path)
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_train = encoder.fit_transform(train_frame[["reviewerID", "itemID"]]).tocsr()
    y_train = train_frame["overall"].to_numpy(dtype=np.float64)
    sparse_matrix_path = Path(sparse_matrix_path)
    sparse_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(sparse_matrix_path, X_train)

    X_valid = None
    y_valid = None
    if validation_data is not None and len(validation_data):
        valid_frame = _interaction_frame(validation_data)
        X_valid = encoder.transform(valid_frame[["reviewerID", "itemID"]]).tocsr()
        y_valid = valid_frame["overall"].to_numpy(dtype=np.float64)

    feature_names = encoder.get_feature_names_out(["reviewerID", "itemID"])
    fm = FactorizationMachine(
        n_factors,
        X_train.shape[1],
        feature_names,
        seed=args.seed,
    )
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        fm.load_checkpoint(checkpoint_path)
    else:
        fm.fit(
            X_train,
            y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            reg=args.reg,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
        )
        fm.save_checkpoint(checkpoint_path)
    return fm

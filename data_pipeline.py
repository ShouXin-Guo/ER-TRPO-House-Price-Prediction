from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


@dataclass
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


class HousingDataProcessor:
    def __init__(
        self,
        target_column: str,
        date_column: Optional[str] = None,
        group_column: Optional[str] = None,
        sequence_length: int = 1,
    ) -> None:
        self.target_column = target_column
        self.date_column = date_column
        self.group_column = group_column
        self.sequence_length = max(1, int(sequence_length))
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self._iqr_bounds: dict[str, tuple[float, float]] = {}

    @staticmethod
    def load_table(path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported dataset format: {suffix}")

    def _basic_clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.target_column not in frame.columns:
            raise ValueError(f"Target column '{self.target_column}' was not found.")
        data = frame.copy()
        data[self.target_column] = pd.to_numeric(data[self.target_column], errors="coerce")
        data = data.loc[np.isfinite(data[self.target_column]) & data[self.target_column].notna()].copy()
        if self.date_column and self.date_column in data.columns:
            data[self.date_column] = pd.to_datetime(data[self.date_column], errors="coerce")
            data = data.loc[data[self.date_column].notna()].copy()
            data["__year"] = data[self.date_column].dt.year
            data["__month"] = data[self.date_column].dt.month
            data["__quarter"] = data[self.date_column].dt.quarter
            data["__time_index"] = data[self.date_column].dt.to_period("M").astype(str)
        return data.reset_index(drop=True)

    def _fit_iqr(self, train: pd.DataFrame) -> None:
        numeric = train.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if c != self.target_column]
        self._iqr_bounds = {}
        for col in numeric + [self.target_column]:
            q1 = float(train[col].quantile(0.25))
            q3 = float(train[col].quantile(0.75))
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                continue
            self._iqr_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    def _apply_iqr(self, data: pd.DataFrame) -> pd.DataFrame:
        mask = np.ones(len(data), dtype=bool)
        for col, (lo, hi) in self._iqr_bounds.items():
            if col in data.columns:
                values = pd.to_numeric(data[col], errors="coerce")
                current = values.isna() | values.between(lo, hi)
                mask &= current.to_numpy()
        return data.loc[mask].reset_index(drop=True)

    def _feature_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [self.target_column]
        if self.date_column and self.date_column in data.columns:
            drop_cols.append(self.date_column)
        features = data.drop(columns=drop_cols, errors="ignore").copy()
        if "__time_index" in features.columns:
            features = features.drop(columns=["__time_index"])
        return features

    def _fit_transformer(self, train_features: pd.DataFrame) -> None:
        self.numeric_columns = train_features.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        self.categorical_columns = [c for c in train_features.columns if c not in self.numeric_columns]

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", MinMaxScaler()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", add_indicator=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.numeric_columns),
                ("cat", categorical_pipe, self.categorical_columns),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.preprocessor.fit(train_features)
        self.feature_names = [str(x) for x in self.preprocessor.get_feature_names_out()]

    def _transform(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor is not fitted.")
        x = self.preprocessor.transform(self._feature_frame(data)).astype(np.float32)
        y = data[self.target_column].to_numpy(dtype=np.float32)
        return x, y

    def _make_sequences(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.sequence_length == 1:
            return x[:, None, :], y
        if len(x) < self.sequence_length:
            raise ValueError("The split contains fewer rows than sequence_length.")
        xs, ys = [], []
        for end in range(self.sequence_length - 1, len(x)):
            start = end - self.sequence_length + 1
            xs.append(x[start : end + 1])
            ys.append(y[end])
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def prepare(self, frame: pd.DataFrame, seed: int) -> DatasetSplit:
        clean = self._basic_clean(frame)
        train, remainder = train_test_split(clean, test_size=0.30, random_state=seed, shuffle=True)
        val, test = train_test_split(remainder, test_size=0.50, random_state=seed, shuffle=True)

        self._fit_iqr(train)
        train = self._apply_iqr(train)
        val = self._apply_iqr(val)
        test = self._apply_iqr(test)

        self._fit_transformer(self._feature_frame(train))
        x_train, y_train = self._transform(train)
        x_val, y_val = self._transform(val)
        x_test, y_test = self._transform(test)

        x_train, y_train = self._make_sequences(x_train, y_train)
        x_val, y_val = self._make_sequences(x_val, y_val)
        x_test, y_test = self._make_sequences(x_test, y_test)
        return DatasetSplit(x_train, y_train, x_val, y_val, x_test, y_test, self.feature_names)


def split_labeled_unlabeled(
    x: np.ndarray,
    y: np.ndarray,
    labeled_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < labeled_fraction < 1:
        raise ValueError("labeled_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    n_labeled = max(2, int(round(len(x) * labeled_fraction)))
    labeled_idx = order[:n_labeled]
    unlabeled_idx = order[n_labeled:]
    return x[labeled_idx], y[labeled_idx], x[unlabeled_idx], y[unlabeled_idx]

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictiveModelResult:
    model_name: str
    feature_columns: list[str]
    metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    decision_table: pd.DataFrame


def _as_numeric_matrix(df: pd.DataFrame, feature_columns: Iterable[str]) -> np.ndarray:
    matrix = df.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any():
        bad_cols = sorted(matrix.columns[matrix.isna().any()].tolist())
        raise ValueError(f"Feature columns contain missing or non-numeric values: {bad_cols}")
    return matrix.to_numpy(dtype=float)


def _standardize_train_test(
    train_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (train_x - mean) / scale, (test_x - mean) / scale, mean, scale


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ y


def _ridge_predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return design @ coef


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_bootstrap_ridge(
    data: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    id_column: str = "molecule",
    test_fraction: float = 0.25,
    n_bootstrap: int = 200,
    alpha: float = 1.0,
    random_state: int = 13,
) -> PredictiveModelResult:
    """Train a transparent uncertainty-aware ridge ensemble.

    The model is intentionally lightweight: it supports public demos and CI without
    proprietary data or heavyweight dependencies, while preserving the same
    validation, uncertainty, and interpretability surfaces used in larger pipelines.
    """

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")

    rng = np.random.default_rng(random_state)
    required = [id_column, target_column, *feature_columns]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    working = data.loc[:, required].dropna().reset_index(drop=True)
    if len(working) < 4:
        raise ValueError("At least four complete rows are required for a train/test split")

    x = _as_numeric_matrix(working, feature_columns)
    y = pd.to_numeric(working[target_column], errors="raise").to_numpy(dtype=float)
    order = rng.permutation(len(working))
    n_test = max(1, int(round(len(working) * test_fraction)))
    n_test = min(n_test, len(working) - 2)
    test_idx = np.sort(order[:n_test])
    train_idx = np.sort(order[n_test:])

    train_x_raw = x[train_idx]
    test_x_raw = x[test_idx]
    train_x, test_x, mean, scale = _standardize_train_test(train_x_raw, test_x_raw)
    train_y = y[train_idx]
    test_y = y[test_idx]

    prediction_matrix = np.zeros((n_bootstrap, len(test_idx)), dtype=float)
    raw_coef_matrix = np.zeros((n_bootstrap, len(feature_columns)), dtype=float)
    standardized_coef_matrix = np.zeros((n_bootstrap, len(feature_columns)), dtype=float)
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, len(train_x), size=len(train_x))
        coef = _ridge_fit(train_x[sample_idx], train_y[sample_idx], alpha=alpha)
        prediction_matrix[i] = _ridge_predict(test_x, coef)
        raw_coef_matrix[i] = coef[1:] / scale
        standardized_coef_matrix[i] = coef[1:]

    pred_mean = prediction_matrix.mean(axis=0)
    pred_std = prediction_matrix.std(axis=0, ddof=1)
    lower = np.quantile(prediction_matrix, 0.05, axis=0)
    upper = np.quantile(prediction_matrix, 0.95, axis=0)
    metrics = regression_metrics(test_y, pred_mean)
    metrics["coverage_90"] = float(np.mean((test_y >= lower) & (test_y <= upper)))
    metrics["mean_prediction_std"] = float(pred_std.mean())

    predictions = pd.DataFrame(
        {
            id_column: working.loc[test_idx, id_column].to_numpy(),
            "actual": test_y,
            "predicted": pred_mean,
            "prediction_std": pred_std,
            "lower_90": lower,
            "upper_90": upper,
            "absolute_error": np.abs(test_y - pred_mean),
        }
    )
    predictions["split"] = "test"

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient_mean": raw_coef_matrix.mean(axis=0),
            "coefficient_std": raw_coef_matrix.std(axis=0, ddof=1),
            "standardized_coefficient_mean": standardized_coef_matrix.mean(axis=0),
            "standardized_coefficient_std": standardized_coef_matrix.std(axis=0, ddof=1),
            "importance": np.abs(standardized_coef_matrix).mean(axis=0),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    decision_table = predictions.sort_values(
        ["prediction_std", "absolute_error"], ascending=[False, False]
    ).reset_index(drop=True)
    decision_table["recommended_action"] = np.where(
        decision_table["prediction_std"] > decision_table["prediction_std"].median(),
        "prioritize confirmatory experiment or higher-fidelity calculation",
        "use prediction as lower-uncertainty screening evidence",
    )

    return PredictiveModelResult(
        model_name="bootstrap_ridge",
        feature_columns=feature_columns,
        metrics=metrics,
        predictions=predictions,
        feature_importance=importance,
        decision_table=decision_table,
    )


def train_random_forest_if_available(
    data: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    id_column: str = "molecule",
    random_state: int = 13,
):
    """Fit a scikit-learn random forest when the optional dependency is installed."""

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install the optional predictive extra to use random-forest benchmarking: "
            "pip install -e .[predictive]"
        ) from exc

    required = [id_column, target_column, *feature_columns]
    working = data.loc[:, required].dropna().reset_index(drop=True)
    x = _as_numeric_matrix(working, feature_columns)
    y = pd.to_numeric(working[target_column], errors="raise").to_numpy(dtype=float)
    train_x, test_x, train_y, test_y, train_ids, test_ids = train_test_split(
        x,
        y,
        working[id_column].to_numpy(),
        test_size=0.25,
        random_state=random_state,
    )
    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=random_state,
    )
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    metrics = regression_metrics(test_y, pred)
    perm = permutation_importance(
        model, test_x, test_y, n_repeats=20, random_state=random_state
    )
    return {
        "model": model,
        "metrics": metrics,
        "predictions": pd.DataFrame(
            {id_column: test_ids, "actual": test_y, "predicted": pred}
        ),
        "feature_importance": pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance", ascending=False, ignore_index=True),
    }

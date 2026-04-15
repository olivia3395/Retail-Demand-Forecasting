from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    num = 2.0 * np.abs(y_true - y_pred)
    mask = denom > 0
    return float(np.mean(num[mask] / denom[mask])) if np.any(mask) else np.nan


def regression_metrics(y_true, y_pred) -> dict:
    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'WAPE': wape(y_true, y_pred),
        'sMAPE': smape(y_true, y_pred),
    }


def group_metrics(df: pd.DataFrame, group_col: str, y_col: str = 'y', pred_col: str = 'prediction') -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(group_col):
        metrics = regression_metrics(grp[y_col], grp[pred_col])
        metrics[group_col] = key
        metrics['n_obs'] = len(grp)
        rows.append(metrics)
    return pd.DataFrame(rows)

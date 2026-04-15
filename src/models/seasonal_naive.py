from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.utils.io import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_validate_seasonal_naive(df: pd.DataFrame, seasonal_lag: int = 7, valid_days: int = 28):
    cutoff = df['date'].max() - pd.Timedelta(days=valid_days - 1)
    train_df = df[df['date'] < cutoff].copy()
    valid_df = df[df['date'] >= cutoff].copy()

    history = train_df[['series_id', 'date', 'y']].copy()
    history = history.rename(columns={'y': 'prediction_source'})

    valid_df['lag_date'] = valid_df['date'] - pd.to_timedelta(seasonal_lag, unit='D')
    valid_df = valid_df.merge(
        history,
        left_on=['series_id', 'lag_date'],
        right_on=['series_id', 'date'],
        how='left',
        suffixes=('', '_hist')
    )
    valid_df['prediction'] = valid_df['prediction_source'].fillna(0.0)
    metrics = regression_metrics(valid_df['y'], valid_df['prediction'])
    return valid_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--artifact_dir', type=str, default='artifacts')
    parser.add_argument('--seasonal_lag', type=int, default=7)
    parser.add_argument('--valid_days', type=int, default=28)
    args = parser.parse_args()

    df = pd.read_csv(args.input_path, parse_dates=['date'])
    preds, metrics = train_validate_seasonal_naive(df, args.seasonal_lag, args.valid_days)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pred_path = artifact_dir / 'predictions' / 'naive_validation_predictions.csv'
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_path, index=False)
    save_json(metrics, artifact_dir / 'metrics' / 'metrics_naive.json')
    logger.info('Naive metrics: %s', metrics)


if __name__ == '__main__':
    main()

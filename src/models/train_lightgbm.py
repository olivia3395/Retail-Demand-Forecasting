from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.utils.io import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


EXCLUDE_COLS = {
    'id', 'series_id', 'date', 'd', 'y', 'wm_yr_wk', 'weekday',
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if df[col].dtype == 'object':
            cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def train_validate_lightgbm(df: pd.DataFrame, params: dict, valid_days: int = 28,
                            max_train_rows: int | None = None):
    cutoff = df['date'].max() - pd.Timedelta(days=valid_days - 1)
    train_df = df[df['date'] < cutoff].copy()
    valid_df = df[df['date'] >= cutoff].copy()

    if max_train_rows is not None and len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=42)

    feature_cols = get_feature_columns(df)
    cat_cols = [c for c in feature_cols if train_df[c].dtype == 'object']
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        valid_df[col] = valid_df[col].astype('category')

    X_train = train_df[feature_cols]
    y_train = train_df['y']
    X_valid = valid_df[feature_cols]
    y_valid = valid_df['y']

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='l1',
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )

    valid_df['prediction'] = np.clip(model.predict(X_valid), 0, None)
    metrics = regression_metrics(valid_df['y'], valid_df['prediction'])

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    return model, valid_df, metrics, feature_importance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--artifact_dir', type=str, default='artifacts')
    parser.add_argument('--valid_days', type=int, default=28)
    parser.add_argument('--max_train_rows', type=int, default=1500000)
    args = parser.parse_args()

    df = pd.read_csv(args.input_path, parse_dates=['date'])
    params = {
        'objective': 'regression',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'min_data_in_leaf': 64,
        'n_estimators': 1200,
        'verbose': -1,
    }

    model, preds, metrics, feature_importance = train_validate_lightgbm(
        df, params=params, valid_days=args.valid_days, max_train_rows=args.max_train_rows
    )

    artifact_dir = Path(args.artifact_dir)
    (artifact_dir / 'models').mkdir(parents=True, exist_ok=True)
    (artifact_dir / 'predictions').mkdir(parents=True, exist_ok=True)
    (artifact_dir / 'metrics').mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_dir / 'models' / 'lightgbm.pkl')
    preds.to_csv(artifact_dir / 'predictions' / 'lightgbm_validation_predictions.csv', index=False)
    feature_importance.to_csv(artifact_dir / 'metrics' / 'lightgbm_feature_importance.csv', index=False)
    save_json(metrics, artifact_dir / 'metrics' / 'metrics_lightgbm.json')
    logger.info('LightGBM metrics: %s', metrics)


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.download_m5 import main as download_main  # noqa: F401
from src.data.download_m5 import download_with_kaggle_api, download_with_kagglehub
from src.data.prepare_m5 import prepare_m5
from src.evaluation.metrics import regression_metrics
from src.features.build_features import build_features
from src.models.seasonal_naive import train_validate_seasonal_naive
from src.models.train_lightgbm import train_validate_lightgbm
from src.models.train_tft import train_validate_tft
from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _download_if_needed(cfg: dict) -> None:
    raw_dir = ensure_dir(cfg['data']['raw_dir'])
    competition = cfg['data']['competition']
    required = {'calendar.csv', 'sales_train_validation.csv', 'sell_prices.csv', 'sample_submission.csv'}
    present = {p.name for p in Path(raw_dir).glob('*.csv')}
    if required.issubset(present):
        logger.info('Raw M5 data already present.')
        return
    ok = download_with_kagglehub(competition, Path(raw_dir))
    if not ok:
        ok = download_with_kaggle_api(competition, Path(raw_dir))
    if not ok:
        raise RuntimeError('Could not download M5 data. Check Kaggle credentials and competition acceptance.')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--models', nargs='*', default=['naive', 'lightgbm', 'tft'])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    artifact_dir = Path(cfg['project']['artifact_dir'])
    ensure_dir(artifact_dir)
    ensure_dir(artifact_dir / 'data')
    ensure_dir(artifact_dir / 'metrics')
    ensure_dir(artifact_dir / 'predictions')
    ensure_dir(artifact_dir / 'models')

    _download_if_needed(cfg)

    processed = prepare_m5(
        raw_dir=cfg['data']['raw_dir'],
        output_path=cfg['data']['processed_path'],
        sample_top_n_series=cfg['data']['sample_top_n_series'],
        min_history_days=cfg['data']['min_history_days'],
    )

    featured = build_features(
        input_path=cfg['data']['processed_path'],
        output_path=cfg['data']['featured_path'],
        lags=cfg['features']['lags'],
        rolling_windows=cfg['features']['rolling_windows'],
        add_price_change=cfg['features']['add_price_change'],
        add_price_ratio_7=cfg['features']['add_price_ratio_7'],
        add_event_flags=cfg['features']['add_event_flags'],
        add_snap_flags=cfg['features']['add_snap_flags'],
        add_calendar_features=cfg['features']['add_calendar_features'],
    )
    featured['date'] = pd.to_datetime(featured['date'])

    valid_days = int(cfg['data']['valid_days'])

    if 'naive' in args.models:
        preds, metrics = train_validate_seasonal_naive(
            featured,
            seasonal_lag=int(cfg['models']['naive']['seasonal_lag']),
            valid_days=valid_days,
        )
        preds.to_csv(artifact_dir / 'predictions' / 'naive_validation_predictions.csv', index=False)
        save_json(metrics, artifact_dir / 'metrics' / 'metrics_naive.json')
        logger.info('Naive metrics: %s', metrics)

    if 'lightgbm' in args.models:
        model, preds, metrics, fi = train_validate_lightgbm(
            featured,
            params=cfg['models']['lightgbm'],
            valid_days=valid_days,
            max_train_rows=int(cfg['data']['max_train_rows_for_lightgbm']),
        )
        import joblib
        joblib.dump(model, artifact_dir / 'models' / 'lightgbm.pkl')
        preds.to_csv(artifact_dir / 'predictions' / 'lightgbm_validation_predictions.csv', index=False)
        fi.to_csv(artifact_dir / 'metrics' / 'lightgbm_feature_importance.csv', index=False)
        save_json(metrics, artifact_dir / 'metrics' / 'metrics_lightgbm.json')
        logger.info('LightGBM metrics: %s', metrics)

    if 'tft' in args.models:
        model, preds, metrics, best_path = train_validate_tft(featured, cfg['models']['tft'], artifact_dir)
        preds.to_csv(artifact_dir / 'predictions' / 'tft_validation_predictions.csv', index=False)
        save_json(metrics, artifact_dir / 'metrics' / 'metrics_tft.json')
        with open(artifact_dir / 'models' / 'tft_best_checkpoint.txt', 'w', encoding='utf-8') as f:
            f.write(str(best_path))
        logger.info('TFT metrics: %s', metrics)


if __name__ == '__main__':
    main()

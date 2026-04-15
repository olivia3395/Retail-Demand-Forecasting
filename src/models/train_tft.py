from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from src.evaluation.metrics import regression_metrics
from src.utils.io import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _prepare_tft_frame(df: pd.DataFrame, top_n_series: int | None = None) -> pd.DataFrame:
    data = df.copy()
    if top_n_series is not None and top_n_series > 0:
        top_ids = data.groupby('series_id', as_index=False)['y'].sum().sort_values('y', ascending=False).head(top_n_series)['series_id']
        data = data[data['series_id'].isin(top_ids)].copy()

    cat_cols = [
        'series_id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
    ]
    for col in cat_cols:
        if col in data.columns:
            data[col] = data[col].fillna('None').astype(str)

    bool_like = ['is_available', 'has_event', 'promo_proxy', 'is_month_start', 'is_month_end']
    for col in bool_like:
        if col in data.columns:
            data[col] = data[col].astype(int)

    return data.sort_values(['series_id', 'time_idx']).reset_index(drop=True)


def train_validate_tft(df: pd.DataFrame, config: dict, artifact_dir: str | Path):
    seed_everything(42, workers=True)
    data = _prepare_tft_frame(df, config.get('top_n_series', 120))

    max_encoder_length = int(config['max_encoder_length'])
    max_prediction_length = int(config['max_prediction_length'])
    training_cutoff = data['time_idx'].max() - max_prediction_length

    static_categoricals = [c for c in ['series_id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] if c in data.columns]
    time_varying_known_categoricals = [
        c for c in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'] if c in data.columns
    ]
    time_varying_known_reals = [
        c for c in ['time_idx', 'price', 'price_change_1', 'price_ratio_7', 'dayofweek', 'weekofyear', 'month', 'day',
                    'snap_CA', 'snap_TX', 'snap_WI', 'has_event', 'promo_proxy', 'is_month_start', 'is_month_end']
        if c in data.columns
    ]
    time_varying_unknown_reals = [
        c for c in ['y', 'lag_1', 'lag_7', 'lag_14', 'lag_28', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
                    'rolling_std_7', 'rolling_std_14', 'rolling_std_28'] if c in data.columns
    ]

    training = TimeSeriesDataSet(
        data[data.time_idx <= training_cutoff],
        time_idx='time_idx',
        target='y',
        group_ids=['series_id'],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=['series_id'], transformation='softplus'),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        min_prediction_idx=training_cutoff + 1,
        stop_randomization=True,
        predict=True,
    )

    train_loader = training.to_dataloader(train=True, batch_size=int(config['batch_size']), num_workers=int(config['num_workers']))
    val_loader = validation.to_dataloader(train=False, batch_size=int(config['batch_size']), num_workers=int(config['num_workers']))

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=float(config['learning_rate']),
        hidden_size=int(config['hidden_size']),
        attention_head_size=int(config['attention_head_size']),
        dropout=float(config['dropout']),
        hidden_continuous_size=int(config['hidden_continuous_size']),
        loss=QuantileLoss(),
        reduce_on_plateau_patience=int(config['reduce_on_plateau_patience']),
    )

    artifact_dir = Path(artifact_dir)
    logger_dir = artifact_dir / 'lightning_logs'
    checkpoint_dir = artifact_dir / 'models' / 'tft_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=int(config['early_stopping_patience']), mode='min'),
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(dirpath=checkpoint_dir, filename='tft-{epoch:02d}-{val_loss:.4f}', monitor='val_loss', mode='min', save_top_k=1),
    ]

    trainer = Trainer(
        max_epochs=int(config['max_epochs']),
        accelerator=config.get('accelerator', 'auto'),
        devices=config.get('devices', 'auto'),
        gradient_clip_val=float(config['gradient_clip_val']),
        limit_train_batches=float(config.get('limit_train_batches', 1.0)),
        limit_val_batches=float(config.get('limit_val_batches', 1.0)),
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(logger_dir), name='tft'),
        enable_model_summary=True,
    )

    trainer.fit(model, train_loader, val_loader)
    best_path = callbacks[-1].best_model_path
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_path)

    preds = best_model.predict(val_loader)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    elif isinstance(preds, tuple):
        preds = preds[0].detach().cpu().numpy()

    idx = validation.decoded_index.reset_index(drop=True).copy()
    if 'time_idx_first_prediction' in idx.columns:
        start_col = 'time_idx_first_prediction'
    elif 'time_idx_first' in idx.columns:
        start_col = 'time_idx_first'
    else:
        raise RuntimeError('Could not find prediction start index in validation.decoded_index')

    pred_rows = []
    for i in range(len(idx)):
        series_id = idx.loc[i, 'series_id'] if 'series_id' in idx.columns else idx.iloc[i, 0]
        start_time_idx = int(idx.loc[i, start_col])
        horizon_values = preds[i]
        if horizon_values.ndim > 1:
            # for quantiles, use median if present else first channel
            q_dim = horizon_values.shape[-1]
            median_idx = q_dim // 2
            horizon_values = horizon_values[:, median_idx]
        for h, pred in enumerate(horizon_values):
            pred_rows.append({'series_id': series_id, 'time_idx': start_time_idx + h, 'prediction': float(pred)})

    pred_df = pd.DataFrame(pred_rows)
    actual_cols = [c for c in ['series_id', 'time_idx', 'date', 'y', 'store_id', 'cat_id', 'dept_id', 'item_id', 'state_id'] if c in data.columns]
    actual_df = data[actual_cols].drop_duplicates(['series_id', 'time_idx'])
    pred_df = pred_df.merge(actual_df, on=['series_id', 'time_idx'], how='left')
    pred_df = pred_df.dropna(subset=['y']).copy()
    pred_df['prediction'] = pred_df['prediction'].clip(lower=0.0)

    metrics = regression_metrics(pred_df['y'], pred_df['prediction'])
    return best_model, pred_df, metrics, best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--artifact_dir', type=str, default='artifacts')
    args = parser.parse_args()

    df = pd.read_csv(args.input_path, parse_dates=['date'])
    config = {
        'max_encoder_length': 56,
        'max_prediction_length': 28,
        'hidden_size': 24,
        'attention_head_size': 4,
        'dropout': 0.1,
        'hidden_continuous_size': 16,
        'batch_size': 128,
        'max_epochs': 8,
        'gradient_clip_val': 0.1,
        'learning_rate': 0.03,
        'accelerator': 'auto',
        'devices': 'auto',
        'num_workers': 0,
        'reduce_on_plateau_patience': 2,
        'early_stopping_patience': 3,
        'top_n_series': 120,
    }
    model, preds, metrics, best_path = train_validate_tft(df, config, args.artifact_dir)

    artifact_dir = Path(args.artifact_dir)
    (artifact_dir / 'predictions').mkdir(parents=True, exist_ok=True)
    (artifact_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    (artifact_dir / 'models').mkdir(parents=True, exist_ok=True)
    preds.to_csv(artifact_dir / 'predictions' / 'tft_validation_predictions.csv', index=False)
    save_json(metrics, artifact_dir / 'metrics' / 'metrics_tft.json')
    logger.info('Saved best TFT checkpoint at %s', best_path)
    logger.info('TFT metrics: %s', metrics)


if __name__ == '__main__':
    main()

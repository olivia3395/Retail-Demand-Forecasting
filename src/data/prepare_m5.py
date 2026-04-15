from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

ID_COLS = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']


def prepare_m5(raw_dir: str | Path, output_path: str | Path, sample_top_n_series: int | None = None,
               min_history_days: int = 200) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    sales = pd.read_csv(raw_dir / 'sales_train_validation.csv')
    calendar = pd.read_csv(raw_dir / 'calendar.csv')
    prices = pd.read_csv(raw_dir / 'sell_prices.csv')

    value_cols = [c for c in sales.columns if c.startswith('d_')]
    logger.info('Melting %s series columns', len(value_cols))

    df = sales.melt(
        id_vars=ID_COLS,
        value_vars=value_cols,
        var_name='d',
        value_name='y'
    )

    df['y'] = df['y'].astype(np.float32)
    calendar['date'] = pd.to_datetime(calendar['date'])
    df = df.merge(calendar, on='d', how='left')
    df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    counts = df.groupby('id').size().reset_index(name='n_obs')
    keep_ids = counts.loc[counts['n_obs'] >= min_history_days, 'id']
    df = df[df['id'].isin(keep_ids)].copy()

    if sample_top_n_series is not None and sample_top_n_series > 0:
        totals = df.groupby('id', as_index=False)['y'].sum().sort_values('y', ascending=False)
        top_ids = totals.head(sample_top_n_series)['id']
        df = df[df['id'].isin(top_ids)].copy()
        logger.info('Keeping top %s series by total demand', sample_top_n_series)

    df['series_id'] = df['id']
    df['time_idx'] = (df['date'] - df['date'].min()).dt.days.astype(int)
    df['price'] = df['sell_price'].astype(np.float32)
    df['is_available'] = df['price'].notna().astype(np.int8)
    df['price'] = df.groupby('id')['price'].ffill().bfill()
    df['price'] = df['price'].fillna(0.0)

    out = Path(output_path)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    logger.info('Saved processed data to %s with shape %s', out, df.shape)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--sample_top_n_series', type=int, default=400)
    parser.add_argument('--min_history_days', type=int, default=200)
    args = parser.parse_args()

    prepare_m5(
        raw_dir=args.raw_dir,
        output_path=args.output_path,
        sample_top_n_series=args.sample_top_n_series,
        min_history_days=args.min_history_days,
    )


if __name__ == '__main__':
    main()

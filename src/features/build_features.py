from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def add_group_lags(df: pd.DataFrame, group_col: str, target_col: str, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag).astype(np.float32)
    return df


def add_group_rolls(df: pd.DataFrame, group_col: str, target_col: str, windows: list[int]) -> pd.DataFrame:
    grp = df.groupby(group_col)[target_col]
    for window in windows:
        df[f'rolling_mean_{window}'] = grp.shift(1).rolling(window).mean().astype(np.float32)
        df[f'rolling_std_{window}'] = grp.shift(1).rolling(window).std().astype(np.float32)
    return df


def build_features(input_path: str | Path, output_path: str | Path, lags: list[int], rolling_windows: list[int],
                   add_price_change: bool = True, add_price_ratio_7: bool = True,
                   add_event_flags: bool = True, add_snap_flags: bool = True,
                   add_calendar_features: bool = True) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=['date'])
    df = df.sort_values(['series_id', 'date']).reset_index(drop=True)

    if add_calendar_features:
        df['dayofweek'] = df['date'].dt.dayofweek.astype(np.int16)
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(np.int16)
        df['month'] = df['date'].dt.month.astype(np.int16)
        df['day'] = df['date'].dt.day.astype(np.int16)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(np.int8)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(np.int8)

    if add_event_flags:
        for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
            if col in df.columns:
                df[col] = df[col].fillna('None').astype(str)
        df['has_event'] = ((df.get('event_name_1', 'None') != 'None') | (df.get('event_name_2', 'None') != 'None')).astype(np.int8)

    if add_snap_flags:
        for col in ['snap_CA', 'snap_TX', 'snap_WI']:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(np.int8)

    df = add_group_lags(df, 'series_id', 'y', lags)
    df = add_group_rolls(df, 'series_id', 'y', rolling_windows)

    if add_price_change:
        df['price_change_1'] = df.groupby('series_id')['price'].pct_change().replace([np.inf, -np.inf], np.nan).astype(np.float32)
    if add_price_ratio_7:
        price_shift_7 = df.groupby('series_id')['price'].shift(7)
        df['price_ratio_7'] = (df['price'] / price_shift_7).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    # simple promo proxy: price drop relative to last week
    df['promo_proxy'] = ((df.get('price_ratio_7', np.nan) < 0.98) & df['price'].gt(0)).astype(np.int8)

    # fill numeric nulls after lagging/rolling
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in num_cols:
        if col in {'y'}:
            continue
        df[col] = df[col].fillna(0)

    out = Path(output_path)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    logger.info('Saved featured data to %s with shape %s', out, df.shape)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--lags', type=int, nargs='+', default=[1, 7, 14, 28])
    parser.add_argument('--rolling_windows', type=int, nargs='+', default=[7, 14, 28])
    args = parser.parse_args()

    build_features(
        input_path=args.input_path,
        output_path=args.output_path,
        lags=args.lags,
        rolling_windows=args.rolling_windows,
    )


if __name__ == '__main__':
    main()

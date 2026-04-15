from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import group_metrics, regression_metrics
from src.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--artifact_dir', type=str, default='artifacts')
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    metrics = regression_metrics(df['y'], df['prediction'])

    artifact_dir = Path(args.artifact_dir)
    model_name = Path(args.predictions).stem.replace('_validation_predictions', '')
    save_json(metrics, artifact_dir / 'metrics' / f'metrics_{model_name}.json')

    if 'cat_id' in df.columns:
        group_metrics(df, 'cat_id').to_csv(
            artifact_dir / 'metrics' / f'metrics_{model_name}_by_category.csv', index=False
        )

    if 'store_id' in df.columns:
        group_metrics(df, 'store_id').to_csv(
            artifact_dir / 'metrics' / f'metrics_{model_name}_by_store.csv', index=False
        )

    print(metrics)


if __name__ == '__main__':
    main()

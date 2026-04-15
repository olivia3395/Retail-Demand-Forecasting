from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_FILES = {
    'calendar.csv',
    'sales_train_validation.csv',
    'sell_prices.csv',
    'sample_submission.csv',
}


def _files_present(raw_dir: Path) -> bool:
    present = {p.name for p in raw_dir.glob('*.csv')}
    return REQUIRED_FILES.issubset(present)


def _extract_zip_files(directory: Path) -> None:
    for zf in directory.glob('*.zip'):
        logger.info('Extracting %s', zf)
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(directory)


def _copy_csvs(src_root: Path, raw_dir: Path) -> None:
    for path in src_root.rglob('*.csv'):
        target = raw_dir / path.name
        if not target.exists():
            shutil.copy2(path, target)


def download_with_kagglehub(competition: str, raw_dir: Path) -> bool:
    try:
        import kagglehub

        logger.info('Trying kagglehub competition download for %s', competition)
        path = Path(kagglehub.competition_download(competition, output_dir=str(raw_dir), force_download=False))
        _extract_zip_files(path)
        _copy_csvs(path, raw_dir)
        return _files_present(raw_dir)
    except Exception as e:
        logger.warning('kagglehub download failed: %s', e)
        return False


def download_with_kaggle_api(competition: str, raw_dir: Path) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        logger.info('Trying Kaggle API competition download for %s', competition)
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(competition=competition, path=str(raw_dir), quiet=False)
        _extract_zip_files(raw_dir)
        return _files_present(raw_dir)
    except Exception as e:
        logger.warning('Kaggle API download failed: %s', e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--competition', type=str, default='m5-forecasting-accuracy')
    parser.add_argument('--raw_dir', type=str, default='artifacts/data/raw')
    args = parser.parse_args()

    raw_dir = ensure_dir(args.raw_dir)

    if _files_present(raw_dir):
        logger.info('M5 files already present in %s', raw_dir)
        return

    ok = download_with_kagglehub(args.competition, raw_dir)
    if not ok:
        ok = download_with_kaggle_api(args.competition, raw_dir)

    if not ok:
        raise RuntimeError(
            'Failed to download M5 dataset. Make sure you accepted the competition rules on Kaggle and configured your Kaggle credentials.'
        )

    logger.info('Download complete. Files available in %s', raw_dir)


if __name__ == '__main__':
    main()

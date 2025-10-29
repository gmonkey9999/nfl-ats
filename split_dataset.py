#!/usr/bin/env python3
"""split_dataset.py

Split the generated ATS dataset into train/test sets.

Modes supported:
- temporal: uses seasons to split; by default uses the latest season as test set
- random: random shuffle split by fraction

Usage examples:
  python split_dataset.py --csv data/nfl_ats_model_dataset_with_players.csv --out_dir data/splits --mode temporal
  python split_dataset.py --csv data/nfl_ats_model_dataset.csv --out_dir data/splits --mode random --test-frac 0.2 --seed 42

Outputs: writes train.csv and test.csv under --out_dir
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List
import json
import hashlib
import sys
from datetime import datetime
import platform
import tarfile
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def _temporal_split(df: pd.DataFrame, test_seasons: List[int] | None = None):
    if 'season' not in df.columns:
        raise ValueError('Temporal split requires a `season` column in the dataset')
    seasons = sorted(df['season'].dropna().unique())
    if len(seasons) == 0:
        raise ValueError('No seasons found in dataset')
    if test_seasons is None:
        test_seasons = [int(seasons[-1])]  # latest season as test by default
    test_seasons = [int(s) for s in test_seasons]
    test = df[df['season'].isin(test_seasons)].copy()
    train = df[~df['season'].isin(test_seasons)].copy()
    return train, test


def _random_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int | None = None):
    if not (0.0 < test_frac < 1.0):
        raise ValueError('test_frac must be between 0 and 1')
    test = df.sample(frac=test_frac, random_state=seed)
    train = df.drop(test.index).copy()
    return train, test


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Split ATS dataset into train/test')
    ap.add_argument('--csv', required=True, help='Input dataset CSV')
    ap.add_argument('--out_dir', required=True, help='Directory to write train.csv and test.csv')
    ap.add_argument('--mode', choices=['temporal', 'random'], default='temporal')
    ap.add_argument('--test-seasons', default=None, help='Comma-separated seasons to use as test (temporal mode)')
    ap.add_argument('--test-frac', type=float, default=0.2, help='Test fraction for random split')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    ap.add_argument('--export-train-by-season', action='store_true', help='Write separate CSVs for each season in the training set (train_<season>.csv)')
    ap.add_argument('--archive-train-by-season', action='store_true', help='Create a compressed tar.gz archive of the train_by_season folder and record checksum in metadata')
    args = ap.parse_args(argv)

    p = Path(args.csv)
    if not p.exists():
        logging.error('Input CSV not found: %s', p)
        return 1
    df = pd.read_csv(p)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'temporal':
        test_seasons = None
        if args.test_seasons:
            test_seasons = [int(s.strip()) for s in args.test_seasons.split(',') if s.strip()]
        train, test = _temporal_split(df, test_seasons=test_seasons)
    else:
        train, test = _random_split(df, test_frac=args.test_frac, seed=args.seed)

    train_path = out_dir / 'train.csv'
    test_path = out_dir / 'test.csv'
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    # write reproducible split metadata
    meta = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'input_csv': str(p.resolve()),
        'input_checksum_sha256': None,
        'mode': args.mode,
        'params': {
            'test_seasons': test_seasons if args.mode == 'temporal' else None,
            'test_frac': args.test_frac if args.mode == 'random' else None,
            'seed': args.seed,
        },
        'counts': {
            'total_rows': len(df),
            'train_rows': len(train),
            'test_rows': len(test),
        },
        'environment': {
            'python': sys.version.replace('\n', ' '),
            'platform': platform.platform(),
            'pandas': pd.__version__,
        }
    }

    try:
        # compute sha256 of the input file for traceability
        h = hashlib.sha256()
        with p.open('rb') as fh:
            for chunk in iter(lambda: fh.read(8192), b''):
                h.update(chunk)
        meta['input_checksum_sha256'] = h.hexdigest()
    except Exception:
        logging.warning('Could not compute checksum for input CSV: %s', p)

    meta_path = out_dir / 'split_meta.json'
    try:
        with meta_path.open('w', encoding='utf8') as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        logging.info('Wrote split metadata: %s', meta_path)
    except Exception:
        logging.exception('Failed to write split metadata: %s', meta_path)

    # Optionally export train set by season
    if args.export_train_by_season:
        by_season_dir = out_dir / 'train_by_season'
        by_season_dir.mkdir(parents=True, exist_ok=True)
        seasons = sorted(train['season'].dropna().astype(int).unique().tolist())
        train_by_season_meta = []
        for s in seasons:
            subset = train[train['season'].astype(int) == int(s)].copy()
            fn = f'train_{int(s)}.csv'
            pth = by_season_dir / fn
            try:
                subset.to_csv(pth, index=False)
                logging.info('Wrote train-by-season: %s (rows=%d)', pth, len(subset))
                train_by_season_meta.append({'season': int(s), 'rows': int(len(subset)), 'path': str(pth.resolve())})
            except Exception:
                logging.exception('Failed to write train-by-season file: %s', pth)

        # add to meta and rewrite
        try:
            meta['train_by_season'] = train_by_season_meta
            # optionally archive the folder
            if args.archive_train_by_season:
                archive_path = out_dir / 'train_by_season.tar.gz'
                try:
                    # create tar.gz of the directory contents
                    with tarfile.open(archive_path, 'w:gz') as tar:
                        # add files with relative names inside archive
                        for p in sorted(by_season_dir.glob('*.csv')):
                            tar.add(p, arcname=p.name)
                    logging.info('Created archive: %s', archive_path)
                    # compute checksum and size
                    h = hashlib.sha256()
                    size = 0
                    with archive_path.open('rb') as fh:
                        for chunk in iter(lambda: fh.read(8192), b''):
                            h.update(chunk)
                            size += len(chunk)
                    meta['train_by_season_archive'] = {
                        'path': str(archive_path.resolve()),
                        'sha256': h.hexdigest(),
                        'size_bytes': size,
                    }
                except Exception:
                    logging.exception('Failed to create archive: %s', archive_path)

            with meta_path.open('w', encoding='utf8') as fh:
                json.dump(meta, fh, indent=2, sort_keys=True)
            logging.info('Updated split metadata with train_by_season entries: %s', meta_path)
        except Exception:
            logging.exception('Failed to update split metadata with train_by_season')

    logging.info('Wrote train: %s (rows=%d)', train_path, len(train))
    logging.info('Wrote test:  %s (rows=%d)', test_path, len(test))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

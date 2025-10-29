#!/usr/bin/env python3
"""validate_dataset.py

Lightweight dataset validator for the pipeline outputs.

Checks performed (defaults):
- file exists and contains >0 rows
- required columns present (default set below; overridable)
- gameday column parseable as datetimes (<= max_bad_date_frac allowed)
- unique `game_id` values
- numeric sanity checks for: travel miles (>=0), home_spread_close parseable
- report columns with missing fraction > threshold

Exit codes: 0=ok, 1=usage/IO error, 2=validation failures

Usage examples:
  /path/to/python -m validate_dataset --csv data/nfl_ats_model_dataset.csv
  /path/to/python -m validate_dataset --csv data/nfl_ats_model_dataset_with_players.csv --max-missing 0.1
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import json
import hashlib

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# Canonical required fields (we accept common alternates via ALIASES)
DEFAULT_REQUIRED = [
    'game_id', 'season', 'week', 'gameday', 'home_team', 'away_team', 'home_spread_close'
]

# Common alternate names for columns we may encounter in pipeline outputs
ALIASES = {
    'game_id': ['game_id', 'gameid', 'id'],
    'gameday': ['gameday', 'gameday_dt', 'date', 'game_date'],
    'home_team': ['home_team', 'home', 'team_home'],
    'away_team': ['away_team', 'away', 'team_away'],
    # spread variants
    'home_spread_close': ['home_spread_close', 'spread', 'spread_line', 'home_spread', 'spread_close'],
    # travel miles
    'home_travel_miles': ['home_travel_miles', 'home_travel', 'home_travel_mile'],
    'away_travel_miles': ['away_travel_miles', 'away_travel', 'away_travel_mile'],
}


def _find_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Return the actual column name present in df for the given canonical name.

    Checks aliases first; returns the first matching actual column name or None.
    """
    # If canonical present as-is, prefer it
    if canonical in df.columns:
        return canonical
    for alt in ALIASES.get(canonical, []):
        if alt in df.columns:
            return alt
    # Not found
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate(df: pd.DataFrame, required: List[str], max_missing_frac: float = 0.2, max_bad_date_frac: float = 0.05) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True

    # Row count
    nrows = len(df)
    if nrows == 0:
        msgs.append('No rows found')
        return False, msgs
    msgs.append(f'Rows: {nrows}')

    # Required columns (map canonical -> actual with aliases)
    missing_cols = []
    mapped = {}
    for c in required:
        actual = _find_column(df, c)
        if actual is None:
            missing_cols.append(c)
        else:
            mapped[c] = actual

    if missing_cols:
        msgs.append(f'Missing required columns (no known aliases found): {missing_cols}')
        ok = False
    else:
        msgs.append('All required columns present (mapped aliases where necessary)')

    # Unique game_id (use mapped name if available)
    game_id_col = mapped.get('game_id') or _find_column(df, 'game_id')
    if game_id_col and game_id_col in df.columns:
        dup_ids = df[game_id_col][df[game_id_col].duplicated()].unique()
        if len(dup_ids) > 0:
            msgs.append(f'Duplicate game_id values found (count={len(dup_ids)} sample={list(dup_ids[:5])})')
            ok = False

    # gameday parseability
    gameday_col = mapped.get('gameday') or _find_column(df, 'gameday')
    if gameday_col and gameday_col in df.columns:
        parsed = pd.to_datetime(df[gameday_col], errors='coerce')
        bad = parsed.isna().sum()
        frac = bad / max(1, nrows)
        msgs.append(f'gameday parse failures: {bad}/{nrows} ({frac:.3f})')
        if frac > max_bad_date_frac:
            msgs.append(f'gameday parse failure fraction {frac:.3f} > allowed {max_bad_date_frac}')
            ok = False

    # Numeric checks: home_travel_miles, away_travel_miles >= 0
    for canonical_col in ('home_travel_miles', 'away_travel_miles'):
        col = mapped.get(canonical_col) or _find_column(df, canonical_col)
        if col and col in df.columns:
            # coerce to numeric
            vals = pd.to_numeric(df[col], errors='coerce')
            neg = (vals < 0).sum()
            nan = vals.isna().sum()
            msgs.append(f'{col}: negatives={neg}, nan={nan}')
            if neg > 0:
                msgs.append(f'Found negative values in {col} (count={neg})')
                ok = False

    # spread parseability
    spread_col = mapped.get('home_spread_close') or _find_column(df, 'home_spread_close')
    if spread_col and spread_col in df.columns:
        s = pd.to_numeric(df[spread_col], errors='coerce')
        nbad = s.isna().sum()
        msgs.append(f'{spread_col} parse failures: {nbad}/{nrows}')
        # don't fail the whole run for some NaNs, but warn if >50%
        if nbad / max(1, nrows) > 0.5:
            msgs.append(f'More than 50% of {spread_col} values failed to parse as numbers')
            ok = False

    # Missingness per column
    miss_cols = []
    for c in df.columns:
        miss = df[c].isna().sum() / max(1, nrows)
        if miss > max_missing_frac:
            miss_cols.append((c, miss))
    if miss_cols:
        msgs.append('Columns exceeding missingness threshold: ' + ', '.join(f'{c}:{frac:.2f}' for c, frac in miss_cols))
        # treat as warning by default

    return ok, msgs


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Validate generated ATS dataset CSVs')
    ap.add_argument('--csv', required=True, help='Path to dataset CSV to validate')
    ap.add_argument('--check-split', action='store_true', help='If set, look for split_meta.json in the CSV directory and validate it against train/test files')
    ap.add_argument('--required-cols', default=None, help='Comma-separated list of required columns (overrides defaults)')
    ap.add_argument('--max-missing', type=float, default=0.20, help='Max allowed fraction missing per column (warning)')
    ap.add_argument('--max-bad-dates', type=float, default=0.05, help='Max allowed fraction of unparseable gameday values')
    args = ap.parse_args(argv)

    p = Path(args.csv)
    if not p.exists():
        logging.error('CSV path does not exist: %s', p)
        return 1
    try:
        df = _read_csv(p)
    except Exception as e:
        logging.error('Failed to read CSV: %s', e)
        return 1

    required = DEFAULT_REQUIRED
    if args.required_cols:
        required = [c.strip() for c in args.required_cols.split(',') if c.strip()]

    ok, msgs = validate(df, required=required, max_missing_frac=args.max_missing, max_bad_date_frac=args.max_bad_dates)
    for m in msgs:
        logging.info(m)

    # Optional: validate split metadata if requested or if a split_meta.json exists next to the CSV
    meta_ok = True
    split_meta_checked = False
    split_meta_path = None
    # Determine candidate directory to look for split_meta.json
    candidate_dir = p if p.is_dir() else p.parent
    candidate_meta = candidate_dir / 'split_meta.json'
    if args.check_split or candidate_meta.exists():
        split_meta_checked = True
        split_meta_path = candidate_meta
        try:
            with candidate_meta.open('r', encoding='utf8') as fh:
                meta = json.load(fh)
        except Exception as e:
            logging.error('Failed to read split_meta.json: %s', e)
            meta_ok = False
            meta = None

        if meta is not None:
            # Basic sanity checks
            required_meta_keys = {'created_at', 'input_csv', 'counts', 'mode', 'params'}
            missing_meta = required_meta_keys - set(meta.keys())
            if missing_meta:
                logging.error('split_meta.json missing keys: %s', missing_meta)
                meta_ok = False
            else:
                # if train/test files exist next to meta, check counts
                train_path = candidate_dir / 'train.csv'
                test_path = candidate_dir / 'test.csv'
                meta_counts = meta.get('counts', {})
                if train_path.exists():
                    try:
                        tdf = pd.read_csv(train_path)
                        if meta_counts.get('train_rows') != len(tdf):
                            logging.error('split_meta.csv train_rows mismatch: meta=%s actual=%s', meta_counts.get('train_rows'), len(tdf))
                            meta_ok = False
                    except Exception as e:
                        logging.error('Failed to read train.csv for split check: %s', e)
                        meta_ok = False
                if test_path.exists():
                    try:
                        tdf = pd.read_csv(test_path)
                        if meta_counts.get('test_rows') != len(tdf):
                            logging.error('split_meta.csv test_rows mismatch: meta=%s actual=%s', meta_counts.get('test_rows'), len(tdf))
                            meta_ok = False
                    except Exception as e:
                        logging.error('Failed to read test.csv for split check: %s', e)
                        meta_ok = False

            # If archive info present in meta, verify checksum matches actual file
            if meta_ok and isinstance(meta.get('train_by_season_archive'), dict):
                arch = meta['train_by_season_archive']
                arch_path = Path(arch.get('path')) if arch.get('path') else None
                expected_sha = arch.get('sha256')
                if arch_path is None or not arch_path.exists():
                    logging.error('Archive referenced in split_meta.json not found: %s', arch.get('path'))
                    meta_ok = False
                elif not expected_sha:
                    logging.error('Archive entry missing sha256 in split_meta.json')
                    meta_ok = False
                else:
                    try:
                        h = hashlib.sha256()
                        with arch_path.open('rb') as fh:
                            for chunk in iter(lambda: fh.read(8192), b''):
                                h.update(chunk)
                        actual = h.hexdigest()
                        if actual != expected_sha:
                            logging.error('Archive SHA256 mismatch: meta=%s actual=%s', expected_sha, actual)
                            meta_ok = False
                        else:
                            logging.info('Archive SHA256 verified: %s', arch_path)
                    except Exception as e:
                        logging.exception('Failed to compute archive checksum: %s', arch_path)
                        meta_ok = False

    if split_meta_checked:
        if meta_ok:
            logging.info('Split metadata check PASSED: %s', split_meta_path)
        else:
            logging.error('Split metadata check FAILED: %s', split_meta_path)

    overall_ok = ok and (meta_ok if split_meta_checked else True)

    if overall_ok:
        logging.info('Validation PASSED: %s', p)
        return 0
    else:
        logging.error('Validation FAILED: %s', p)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

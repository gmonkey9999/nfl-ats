#!/usr/bin/env python3
"""
build_nfl_ats_dataset.py

Builds a minimal ATS modeling dataset. This script prefers CSV fallbacks
(`--schedules_csv` and `--lines_csv`) but can attempt to use `nfl_data_py`
when `--seasons` is provided.

The goal here is to provide a small, robust implementation sufficient for
local development and for the `run_pipeline.py` runner.

Usage examples
--------------
python build_nfl_ats_dataset.py --schedules_csv schedules.csv --lines_csv lines.csv --out_csv data/nfl_ats_model_dataset.csv
python build_nfl_ats_dataset.py --seasons 2018,2019 --out_csv data/nfl_ats_model_dataset.csv
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def parse_seasons(s: str) -> List[int]:
    """Parse a seasons string like '2018-2020' or '2018,2019' into a list of ints."""
    if not s:
        return []
    s = s.strip()
    if '-' in s:
        a, b = s.split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',') if x.strip()]


def try_load_via_nfl_data_py(years: List[int]) -> pd.DataFrame | None:
    """Attempt to fetch schedules/lines via nfl_data_py if available.

    This function is conservative: it only proceeds if `nfl_data_py` exposes a
    callable `import_schedules` or `import_schedules_and_lines`. If not
    available, it returns None so the caller can fall back to CSVs.
    """
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception:
        logging.info("nfl_data_py not available in this environment")
        return None

    # Try a few possible function names defensively
    func = getattr(nfl, 'import_schedules', None) or getattr(nfl, 'import_schedules_and_lines', None)
    if not callable(func):
        logging.info("nfl_data_py installed but no compatible schedule import function found")
        return None

    logging.info("Fetching schedules/lines via nfl_data_py for seasons: %s", years)
    try:
        # Many versions accept a list/iterable of years or a single year.
        df = func(years)
        return pd.DataFrame(df)
    except Exception as e:  # pragma: no cover - environmental
        logging.error("Failed to fetch schedules via nfl_data_py: %s", e)
        return None


def build_from_csvs(schedules_csv: Path, lines_csv: Path) -> pd.DataFrame:
    """Load schedules and lines CSVs and merge them into a minimal ATS dataset.

    This function attempts a best-effort merge using `game_id` if present,
    otherwise falls back to (season, week, home_team, away_team) join keys.
    """
    logging.info("Reading schedules CSV: %s", schedules_csv)
    schedules = pd.read_csv(schedules_csv)
    logging.info("Reading lines CSV: %s", lines_csv)
    lines = pd.read_csv(lines_csv)

    # Prefer to merge on game_id if available
    if 'game_id' in schedules.columns and 'game_id' in lines.columns:
        df = schedules.merge(lines, on='game_id', how='left', suffixes=('_sch', '_lines'))
    else:
        # Try common column name variants
        left_cols = []
        for c in ('season', 'week', 'home_team', 'away_team'):
            if c in schedules.columns:
                left_cols.append(c)
        right_cols = []
        for c in ('season', 'week', 'home_team', 'away_team'):
            if c in lines.columns:
                right_cols.append(c)

        common = [c for c in left_cols if c in right_cols]
        if common:
            df = schedules.merge(lines, on=common, how='left', suffixes=('_sch', '_lines'))
        else:
            # As a last resort, concatenate columns (best-effort)
            logging.warning('Could not find common join keys; concatenating schedules and lines side-by-side')
            df = pd.concat([schedules.reset_index(drop=True), lines.reset_index(drop=True)], axis=1)

    # Produce a minimal, consistent set of columns commonly used downstream
    out_cols = {}
    # map common names to canonical names if present
    if 'game_id' in df.columns:
        out_cols['game_id'] = df['game_id']
    if 'season' in df.columns:
        out_cols['season'] = df['season']
    if 'week' in df.columns:
        out_cols['week'] = df['week']
    # home/away team columns - try several common names
    for k in ('home_team', 'home', 'home_tm', 'home_team_abbr'):
        if k in df.columns:
            out_cols['home_team'] = df[k]
            break
    for k in ('away_team', 'away', 'away_tm', 'away_team_abbr'):
        if k in df.columns:
            out_cols['away_team'] = df[k]
            break

    # betting lines
    for k in ('spread', 'line', 'closing_spread'):
        if k in df.columns:
            out_cols['spread'] = df[k]
            break
    for k in ('home_moneyline', 'home_ml'):
        if k in df.columns:
            out_cols['home_moneyline'] = df[k]
            break
    for k in ('away_moneyline', 'away_ml'):
        if k in df.columns:
            out_cols['away_moneyline'] = df[k]
            break

    # scores if present
    for k in ('home_score', 'home_pts'):
        if k in df.columns:
            out_cols['home_score'] = df[k]
            break
    for k in ('away_score', 'away_pts'):
        if k in df.columns:
            out_cols['away_score'] = df[k]
            break

    if not out_cols:
        logging.error('Merged data contains none of the expected columns; cannot build ATS dataset')
        raise SystemExit(1)

    out = pd.DataFrame(out_cols)
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Build core NFL ATS dataset (minimal implementation)')
    ap.add_argument('--seasons', type=str, default=None, help="e.g. '2018-2020' or '2018,2019'")
    ap.add_argument('--schedules_csv', type=str, default=None, help='Path to schedules CSV (optional)')
    ap.add_argument('--lines_csv', type=str, default=None, help='Path to betting lines CSV (optional)')
    ap.add_argument('--out_csv', type=str, required=True, help='Where to write the ATS CSV')

    args = ap.parse_args(argv)

    out_csv = Path(args.out_csv)
    years = parse_seasons(args.seasons) if args.seasons else []

    df: pd.DataFrame | None = None

    # Prefer CSV fallbacks if both provided
    if args.schedules_csv and args.lines_csv:
        df = build_from_csvs(Path(args.schedules_csv), Path(args.lines_csv))
    elif years:
        df = try_load_via_nfl_data_py(years)
        if df is None:
            logging.error('No CSV fallbacks provided and nfl_data_py could not fetch schedules.\n'
                          'Either install nfl_data_py in your environment or provide --schedules_csv and --lines_csv')
            return 2
    else:
        logging.error('Either --seasons or both --schedules_csv and --lines_csv must be provided')
        return 2

    if df is None:
        logging.error('Failed to build ATS dataset')
        return 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logging.info('Writing ATS dataset to %s', out_csv)
    df.to_csv(out_csv, index=False)
    logging.info('Wrote %d rows', len(df))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

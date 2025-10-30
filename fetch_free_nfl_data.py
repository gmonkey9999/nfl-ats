#!/usr/bin/env python3
"""
fetch_free_nfl_data.py

Pulls a bundle of **free/public** NFL datasets to seed or retrain your ATS model.
Designed to be idempotent, verbose (logging), and easy to schedule (launchd/CI).

What it fetches (modular; all optional):
1) nfl_data_py
   - schedules (with spreads/totals when available)
   - play-by-play (PBP)
2) nflverse stadium metadata (roof/surface/coords) via raw GitHub CSV
3) FiveThirtyEight NFL Elo ratings (live CSV)
4) (Optional) Kaggle: NFL Scores & Betting Data (if locally provided or via Kaggle API)
5) (Optional) The Odds API historical odds (requires API key; off by default)

Outputs (default):
  data/raw/
    schedules_YYYY.csv
    pbp_YYYY.parquet (or csv if --pbp-format csv)
    stadiums.csv
    fte_elo_latest.csv
    kaggle_nfl_scores_betting.csv (if provided)

Usage examples
--------------
# Minimal (latest season autodetected as current year)
python fetch_free_nfl_data.py --seasons 2018-2025

# CSV for schedules, Parquet for pbp (recommended for size/speed)
python fetch_free_nfl_data.py --seasons 2018-2025 --pbp-format parquet

# Include Kaggle file (already downloaded via web) and save a normalized copy
python fetch_free_nfl_data.py --kaggle-csv ~/Downloads/nfl-scores-and-betting.csv

# Try to pull The Odds API (needs key); writes data/raw/odds_YYYY.jsonl
python fetch_free_nfl_data.py --odds-api-key $THE_ODDS_API_KEY --sportsbook pinnacle --seasons 2024-2025

Notes
-----
- Requires: pip install nfl-data-py pandas numpy requests pyarrow (for parquet)
- Kaggle: either manually download the dataset CSV and pass --kaggle-csv, or install kaggle CLI and adapt the loader.
- Safe to re-run; files are overwritten unless --skip-existing is set.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ----------------------------
# Helpers
# ----------------------------

def parse_seasons(spec: Optional[str]) -> List[int]:
    if not spec:
        from datetime import datetime
        return [datetime.now().year]
    s = spec.strip()
    if '-' in s:
        a, b = s.split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',') if x.strip()]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path, fmt: str = 'csv') -> None:
    if fmt == 'csv':
        df.to_csv(path, index=False)
    elif fmt == 'parquet':
        try:
            import pyarrow  # noqa: F401
        except Exception:
            logging.warning('pyarrow not installed; falling back to CSV for %s', path.name)
            df.to_csv(path.with_suffix('.csv'), index=False)
            return
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f'Unknown format: {fmt}')
    logging.info('Wrote %s (%d rows)', path, len(df))


# ----------------------------
# Fetchers
# ----------------------------

def fetch_schedules(years: List[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    logging.info('Fetching schedules for seasons %s', years)
    df = nfl.import_schedules(years)
    return pd.DataFrame(df)


def fetch_pbp(years: List[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    logging.info('Fetching play-by-play for seasons %s', years)
    df = nfl.import_pbp_data(years)
    return pd.DataFrame(df)


def fetch_stadiums() -> pd.DataFrame:
    import requests
    url = 'https://raw.githubusercontent.com/nflverse/nflverse-data/master/data/stadiums.csv'
    logging.info('Downloading stadiums metadata from nflverse: %s', url)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(r.text)) if hasattr(pd.compat, 'StringIO') else pd.read_csv(Path(pd.io.common.StringIO(r.text)))
    # Some pandas builds prefer io.StringIO
    return df


def fetch_fte_elo() -> pd.DataFrame:
    import requests
    url = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo_latest.csv'
    logging.info('Downloading FiveThirtyEight Elo: %s', url)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))


def normalize_kaggle_csv(src_csv: Path) -> pd.DataFrame:
    logging.info('Normalizing Kaggle scores/odds CSV: %s', src_csv)
    df = pd.read_csv(src_csv)
    # Light rename for common fields if present
    rename = {
        'home_team': 'home_team',
        'away_team': 'away_team',
        'schedule_season': 'season',
        'schedule_week': 'week',
        'spread_favorite': 'spread_favorite',
        'spread_line': 'spread_line',
        'team_home': 'home_team',
        'team_away': 'away_team',
        'score_home': 'home_score',
        'score_away': 'away_score',
        'over_under_line': 'over_under_line',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def fetch_odds_for_year(year: int, api_key: str, sportsbook: str = 'pinnacle') -> List[dict]:
    """Fetch historical odds snapshots for a season.
    Note: The Odds API requires an API key; free tier may limit history.
    Writes JSONL per season for flexibility.
    """
    import requests
    # Example endpoint (you may need to adjust based on provider docs)
    # Here we query upcoming + recent to build a snapshot; true full history may require paid tier.
    url = f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds-history/?regions=us&markets=h2h,spreads,totals&apiKey={api_key}'
    logging.info('Fetching odds history snapshot for %s via %s', year, sportsbook)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        # Attach year tag just in case
        for d in data:
            d['year_tag'] = year
        return data
    except Exception as e:
        logging.warning('Odds fetch failed for %s: %s', year, e)
        return []


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description='Fetch a bundle of free NFL datasets for ATS modeling')
    ap.add_argument('--seasons', type=str, default=None, help="e.g. '2018-2025' or '2018,2019,2020'")
    ap.add_argument('--outdir', type=str, default='data/raw', help='Where to write outputs')
    ap.add_argument('--pbp-format', choices=['csv','parquet'], default='parquet')
    ap.add_argument('--skip-existing', action='store_true', help='Do not overwrite files that already exist')
    ap.add_argument('--kaggle-csv', type=str, default=None, help='Path to local Kaggle NFL scores/odds CSV to normalize + copy')
    ap.add_argument('--odds-api-key', type=str, default=None, help='The Odds API key (optional)')
    ap.add_argument('--sportsbook', type=str, default='pinnacle', help='Sportsbook tag for odds snapshots (if using API)')
    args = ap.parse_args()

    years = parse_seasons(args.seasons)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # 1) Schedules
    try:
        sched = fetch_schedules(years)
        # write per-season files for easier diffs
        for y in years:
            part = sched[sched.get('season', pd.Series([])).astype('Int64') == y]
            if part.empty:
                continue
            p = outdir / f'schedules_{y}.csv'
            if args.skip_existing and p.exists():
                logging.info('Skip existing %s', p)
            else:
                save_df(part, p, fmt='csv')
    except Exception as e:
        logging.error('Schedules fetch failed: %s', e)

    # 2) Play-by-play
    try:
        pbp = fetch_pbp(years)
        for y in years:
            part = pbp[pbp.get('season', pd.Series([])).astype('Int64') == y]
            if part.empty:
                continue
            p = outdir / f'pbp_{y}.{"parquet" if args.pbp_format=="parquet" else "csv"}'
            if args.skip_existing and p.exists():
                logging.info('Skip existing %s', p)
            else:
                save_df(part, p, fmt=args.pbp_format)
    except Exception as e:
        logging.error('PBP fetch failed: %s', e)

    # 3) Stadiums
    try:
        stadiums = fetch_stadiums()
        p = outdir / 'stadiums.csv'
        if args.skip_existing and p.exists():
            logging.info('Skip existing %s', p)
        else:
            save_df(stadiums, p, fmt='csv')
    except Exception as e:
        logging.error('Stadiums download failed: %s', e)

    # 4) FiveThirtyEight Elo
    try:
        elo = fetch_fte_elo()
        p = outdir / 'fte_elo_latest.csv'
        if args.skip_existing and p.exists():
            logging.info('Skip existing %s', p)
        else:
            save_df(elo, p, fmt='csv')
    except Exception as e:
        logging.error('FiveThirtyEight Elo download failed: %s', e)

    # 5) Kaggle (optional)
    if args.kaggle_csv:
        try:
            kag = normalize_kaggle_csv(Path(args.kaggle_csv))
            p = outdir / 'kaggle_nfl_scores_betting.csv'
            if args.skip_existing and p.exists():
                logging.info('Skip existing %s', p)
            else:
                save_df(kag, p, fmt='csv')
        except Exception as e:
            logging.error('Kaggle CSV normalization failed: %s', e)

    # 6) The Odds API (optional, best-effort)
    if args.odds_api_key:
        all_rows = []
        for y in years:
            rows = fetch_odds_for_year(y, api_key=args.odds_api_key, sportsbook=args.sportsbook)
            all_rows.extend(rows)
        if all_rows:
            p = outdir / 'odds_history.jsonl'
            if not (args.skip_existing and p.exists()):
                with p.open('w', encoding='utf-8') as f:
                    for row in all_rows:
                        f.write(json.dumps(row) + '\n')
                logging.info('Wrote %s (%d snapshots)', p, len(all_rows))

    # Catalog (simple manifest)
    manifest = {
        'years': years,
        'outputs': sorted([str(p.name) for p in outdir.glob('*')]),
    }
    with (outdir / 'data_catalog.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    logging.info('Wrote data catalog with %d items', len(manifest['outputs']))


if __name__ == '__main__':
    main()

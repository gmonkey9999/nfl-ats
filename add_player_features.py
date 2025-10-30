#!/usr/bin/env python3
"""
add_player_features.py

Adds player-level QB features (EPA, completion %, INTs, sacks, success rate, etc.) to the
ATS modeling dataset produced by build_nfl_ats_dataset.py.

Key update:
- **No hard dependency on `nfl_data_py` at runtime.** If `nfl_data_py` is unavailable in your
  environment, pass `--pbp_csv <path>` with previously exported play-by-play data to run the script.
  Otherwise, install it with: `pip install nfl_data_py`.

Inputs
------
- --in_csv      : Path to nfl_ats_model_dataset.csv (from your existing pipeline)
- --out_csv     : Path to write the enriched dataset (default: nfl_ats_model_dataset_with_players.csv)
- --min_dropbacks : Minimum dropbacks to consider a QB "primary" for a game (default: 10)
- --rolls       : Comma-separated list of rolling windows to compute (team- and QB-level), e.g. "3,5" (default: 3,5)
- --pbp_csv     : Optional path to a CSV containing play-by-play data (fallback when `nfl_data_py` is not available)
- --run_tests   : If provided, runs a small test suite against synthetic data and exits.

What it does
------------
1) Detects seasons present in your dataset.
2) Loads play-by-play for those seasons either via `nfl_data_py.import_pbp_data(years)` or from `--pbp_csv`.
3) Identifies the primary QB for each team in each game using dropbacks (attempts + sacks + scrambles).
4) Aggregates game-level QB features for the primary QB.
5) Builds *pre-game* rolling features for each team and for each primary QB (to avoid leakage via `shift(1)`).
6) Merges features into the ATS dataset for both home and away teams.

Result
------
Writes an enriched CSV with QB_* features for home and away, plus rolling variants (QB_*_L3, QB_*_L5, TEAM_*_L3, TEAM_*_L5).

Notes
-----
- This script is defensive about column availability across nflfastR/nfl_data_py versions. If a column is missing,
  it will compute the feature from alternatives when possible or skip gracefully.
- Rolling features are computed using season/week ordering. Weeks must be integers and games uniquely
  identified by game_id.
"""
from __future__ import annotations
import argparse
import logging
import sys
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Try to import nfl_data_py, but make it optional with a clear fallback
HAVE_NFL_DATA_PY = False
try:
    import nfl_data_py as nfl  # type: ignore
    HAVE_NFL_DATA_PY = True
except Exception:
    logging.warning(
        "`nfl_data_py` not found. You can either install it (pip install nfl_data_py) "
        "or provide play-by-play via --pbp_csv."
    )


# ------------------------------
# Utility helpers
# ------------------------------

def _exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _qb_dropback_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for QB dropbacks (pass attempt | sack | scramble).
    If a precomputed `qb_dropback` column exists, prefer it.
    """
    if _exists(df, 'qb_dropback'):
        return df['qb_dropback'].astype(bool)
    parts = []
    for c in ['pass', 'pass_attempt', 'sack', 'qb_scramble', 'scramble']:
        if _exists(df, c):
            parts.append(df[c].fillna(0).astype(int) > 0)
    if parts:
        return np.logical_or.reduce(parts)
    if _exists(df, 'pass'):
        return df['pass'].astype(bool)
    return pd.Series(False, index=df.index)


def _bool_col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name].fillna(0).astype(int) if _exists(df, name) else pd.Series(0, index=df.index)


def _num_col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors='coerce').fillna(0) if _exists(df, name) else pd.Series(0.0, index=df.index)


# ------------------------------
# Feature builders
# ------------------------------

def build_qb_game_features(pbp: pd.DataFrame, min_dropbacks: int = 10) -> pd.DataFrame:
    """Aggregate primary-QB per-game features from play-by-play.

    Returns one row per (game_id, team) with identified primary QB and features.
    """
    logging.info("Building primary QB game-level features...")

    needed_cols = ['game_id', 'season', 'week', 'posteam', 'passer_id', 'passer', 'yards_gained']
    for c in needed_cols:
        if c not in pbp.columns:
            pbp[c] = np.nan

    pbp = pbp.copy()

    # Normalize season/week types
    pbp['season'] = pd.to_numeric(pbp['season'], errors='coerce').astype('Int64')
    pbp['week'] = pd.to_numeric(pbp['week'], errors='coerce').astype('Int64')

    # Identify dropbacks & keep offensive plays with a posteam
    pbp['is_dropback'] = _qb_dropback_mask(pbp)
    pbp = pbp[~pbp['posteam'].isna()].copy()

    # Fill QB identifiers when missing
    if 'qb_id' in pbp.columns:
        pbp['passer_id'] = pbp['passer_id'].fillna(pbp['qb_id'])
    if 'qb' in pbp.columns and 'passer' in pbp.columns:
        pbp['passer'] = pbp['passer'].fillna(pbp['qb'])

    # Per-play helpers
    pbp['comp'] = _bool_col(pbp, 'complete_pass')
    pbp['att'] = _bool_col(pbp, 'pass_attempt')
    pbp['int'] = _bool_col(pbp, 'interception')
    pbp['sack'] = _bool_col(pbp, 'sack')
    pbp['scramble'] = _bool_col(pbp, 'qb_scramble') if _exists(pbp, 'qb_scramble') else _bool_col(pbp, 'scramble')
    pbp['dropback'] = pbp['is_dropback'].astype(int)

    # EPA columns (prefer qb_epa, else epa on dropbacks)
    if _exists(pbp, 'qb_epa'):
        pbp['qb_epa_play'] = _num_col(pbp, 'qb_epa')
    elif _exists(pbp, 'epa'):
        pbp['qb_epa_play'] = _num_col(pbp, 'epa')
    else:
        pbp['qb_epa_play'] = 0.0

    # CPOE if available
    pbp['cpoe_play'] = _num_col(pbp, 'cpoe') if _exists(pbp, 'cpoe') else np.nan

    # Passing yards (prefer explicit column)
    if _exists(pbp, 'passing_yards'):
        pbp['pass_yds'] = _num_col(pbp, 'passing_yards')
    else:
        pbp['pass_yds'] = np.where(pbp['att'] == 1, _num_col(pbp, 'yards_gained'), 0.0)

    # Success indicator (EPA > 0)
    pbp['success'] = (pbp['qb_epa_play'] > 0).astype(int)

    # Aggregate by game/team/passer
    gb = pbp.groupby(['game_id', 'season', 'week', 'posteam', 'passer_id', 'passer'], dropna=False)
    agg = gb.agg(
        dropbacks=('dropback', 'sum'),
        attempts=('att', 'sum'),
        completions=('comp', 'sum'),
        interceptions=('int', 'sum'),
        sacks=('sack', 'sum'),
        scrambles=('scramble', 'sum'),
        pass_yards=('pass_yds', 'sum'),
        qb_epa_total=('qb_epa_play', 'sum'),
        qb_epa_per_play=('qb_epa_play', 'mean'),
        success_rate=('success', 'mean'),
        cpoe_mean=('cpoe_play', 'mean'),
    ).reset_index()

    # Choose primary QB by dropbacks
    agg = agg.sort_values(['game_id', 'posteam', 'dropbacks'], ascending=[True, True, False])
    primary = agg.groupby(['game_id', 'posteam'], as_index=False).head(1).copy()

    # Sample size flag
    primary['primary_qb_low_sample'] = (primary['dropbacks'] < int(min_dropbacks)).astype(int)

    # Derived rates
    primary['comp_pct'] = np.divide(primary['completions'], primary['attempts'].replace(0, np.nan)).fillna(0.0)
    primary['int_rate'] = np.divide(primary['interceptions'], primary['attempts'].replace(0, np.nan)).fillna(0.0)
    primary['sack_rate'] = np.divide(primary['sacks'], primary['dropbacks'].replace(0, np.nan)).fillna(0.0)
    primary['ypa'] = np.divide(primary['pass_yards'], primary['attempts'].replace(0, np.nan)).fillna(0.0)

    # Rename posteam -> team for merge clarity
    primary = primary.rename(columns={'posteam': 'team'})

    return primary


def _add_pre_game_rollings(
    df_team_games: pd.DataFrame,
    key_cols: List[str],
    sort_cols: List[str],
    feature_cols: List[str],
    windows: List[int],
    prefix: str,
) -> pd.DataFrame:
    """Compute pre-game rolling means for specified windows with leakage-safe shift(1)."""
    logging.info(f"Computing rolling averages for {prefix} features with windows: {windows}")
    df = df_team_games.sort_values(sort_cols).copy()
    for feat in feature_cols:
        df[f'{feat}__shift1'] = df.groupby(key_cols, group_keys=False)[feat].shift(1)
    for w in windows:
        for feat in feature_cols:
            name = f'{prefix}{feat}_L{w}'
            df[name] = (
                df.groupby(key_cols, group_keys=False)[f'{feat}__shift1']
                .rolling(w, min_periods=1)
                .mean()
                .reset_index(level=key_cols, drop=True)
            )
    drop_helpers = [c for c in df.columns if c.endswith('__shift1')]
    df = df.drop(columns=drop_helpers)
    return df


def add_rolling_features(primary_qb_games: pd.DataFrame, windows: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pre-game rolling features for teams and primary QBs.

    Returns (team_roll_df, qb_roll_df) where each dataframe contains one row
    per game and the rolling features prefixed with TEAM_ or QB_. This is a
    simplified, leakage-safe implementation using shift(1) before rolling.
    """
    logging.info("Generating team-level and QB-level rolling features...")
    base_feats = ['qb_epa_per_play', 'success_rate', 'comp_pct', 'int_rate', 'sack_rate', 'ypa', 'dropbacks']

    # TEAM-level rolling: group by team
    team_games = primary_qb_games.copy()
    team_games = team_games.sort_values(['team', 'season', 'week'])
    team_roll = _add_pre_game_rollings(team_games, ['team'], ['team', 'season', 'week'], base_feats, windows, prefix='TEAM_')

    # QB-level rolling: group by passer_id (primary QB identifier)
    qb_games = primary_qb_games.copy()
    # ensure passer_id exists
    if 'passer_id' not in qb_games.columns:
        qb_games['passer_id'] = qb_games.get('passer_id', qb_games.get('passer', pd.NA))
    qb_games = qb_games.sort_values(['passer_id', 'season', 'week'])
    qb_roll = _add_pre_game_rollings(qb_games, ['passer_id'], ['passer_id', 'season', 'week'], base_feats, windows, prefix='QB_')

    return team_roll, qb_roll


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Add QB player features to ATS dataset (simplified)')
    ap.add_argument('--in_csv', type=str, required=True, help='Input ATS CSV')
    ap.add_argument('--out_csv', type=str, required=True, help='Output enriched CSV')
    ap.add_argument('--min_dropbacks', type=int, default=10)
    ap.add_argument('--rolls', type=str, default='3,5')
    ap.add_argument('--pbp_csv', type=str, default=None)

    args = ap.parse_args(argv)

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)

    ats = pd.read_csv(in_csv)

    # Load play-by-play either from CSV fallback or nfl_data_py if available
    pbp: pd.DataFrame | None = None
    if args.pbp_csv:
        logging.info('Loading play-by-play from %s', args.pbp_csv)
        pbp = pd.read_csv(args.pbp_csv)
    else:
        if HAVE_NFL_DATA_PY:
            # Try to import pbp for seasons found in ATS
            seasons = sorted(pd.to_numeric(ats['season'], errors='coerce').dropna().unique())
            try:
                func = getattr(nfl, 'import_pbp_data', None) or getattr(nfl, 'import_pbp', None)
                if callable(func):
                    logging.info('Fetching play-by-play via nfl_data_py for seasons: %s', seasons)
                    pbp = pd.DataFrame(func(seasons))
                else:
                    logging.warning('nfl_data_py present but no compatible pbp import function; proceeding with placeholders')
            except Exception as e:
                logging.warning('Failed to fetch pbp via nfl_data_py: %s', e)

    if pbp is None:
        logging.info('No play-by-play available; creating placeholder QB features')
        # Create placeholder columns and write
        ats['QB_home'] = pd.NA
        ats['QB_away'] = pd.NA
        ats['QB_home_dropbacks'] = pd.NA
        ats['QB_away_dropbacks'] = pd.NA
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        ats.to_csv(out_csv, index=False)
        logging.info('Wrote enriched CSV with placeholder QB fields to %s', out_csv)
        return 0

    # Build primary-QB per-game features from pbp
    primary_qb_games = build_qb_game_features(pbp, min_dropbacks=args.min_dropbacks)

    # Compute rolling windows
    windows = [int(x) for x in args.rolls.split(',') if x]
    team_roll, qb_roll = add_rolling_features(primary_qb_games, windows)

    # Merge TEAM and QB features into ATS for home/away
    out = ats.copy()

    # Team-level merges
    if 'home_team' in out.columns:
        team_h = team_roll.rename(columns={'team': 'home_team'})
        # Prefix TEAM_ rolling columns with 'home_' so resulting column names
        # are canonical (e.g. home_TEAM_qb_epa_per_play_L3)
        team_h = team_h.rename(columns={c: f'home_{c}' for c in team_h.columns if c.startswith('TEAM_')})
        out = out.merge(team_h, on=['game_id', 'home_team', 'season'], how='left')
    if 'away_team' in out.columns:
        team_a = team_roll.rename(columns={'team': 'away_team'})
        team_a = team_a.rename(columns={c: f'away_{c}' for c in team_a.columns if c.startswith('TEAM_')})
        out = out.merge(team_a, on=['game_id', 'away_team', 'season'], how='left')

    # QB-level merges (primary QB for team/game)
    if 'home_team' in out.columns:
        qb_h = qb_roll.rename(columns={'team': 'home_team', 'passer_id': 'home_passer_id', 'passer': 'home_passer'})
        # Rename primary-QB per-game columns to canonical home_* names to avoid
        # suffix collisions (e.g. home_qb_epa_per_play)
        rename_map_h = {}
        if 'qb_epa_per_play' in qb_h.columns:
            rename_map_h['qb_epa_per_play'] = 'home_qb_epa_per_play'
        if 'success_rate' in qb_h.columns:
            rename_map_h['success_rate'] = 'home_qb_success_rate'
        # Also prefix any QB_ rolling columns with home_ (optional)
        for c in qb_h.columns:
            if c.startswith('QB_'):
                rename_map_h[c] = f'home_{c}'
        qb_h = qb_h.rename(columns=rename_map_h)
        out = out.merge(qb_h, on=['game_id', 'home_team', 'season'], how='left')
    if 'away_team' in out.columns:
        qb_a = qb_roll.rename(columns={'team': 'away_team', 'passer_id': 'away_passer_id', 'passer': 'away_passer'})
        rename_map_a = {}
        if 'qb_epa_per_play' in qb_a.columns:
            rename_map_a['qb_epa_per_play'] = 'away_qb_epa_per_play'
        if 'success_rate' in qb_a.columns:
            rename_map_a['success_rate'] = 'away_qb_success_rate'
        for c in qb_a.columns:
            if c.startswith('QB_'):
                rename_map_a[c] = f'away_{c}'
        qb_a = qb_a.rename(columns=rename_map_a)
        out = out.merge(qb_a, on=['game_id', 'away_team', 'season'], how='left')

    # Clean up legacy duplicate-suffix columns before writing so downstream
    # consumers see canonical column names only (coalesce *_x/_y and .1 variants).
    def _coalesce_and_clean(df):
        import re
        cols = df.columns.tolist()
        bases = set()
        # compute canonical base by stripping known suffixes
        for c in cols:
            base = re.sub(r'(_x|_y|\.1)$', '', c)
            bases.add(base)

        for base in bases:
            # candidate variants in preferred order
            candidates = [
                base,
                f"{base}_x",
                f"{base}_y",
                f"{base}_x.1",
                f"{base}_y.1",
                f"{base}.1",
            ]
            present = [c for c in candidates if c in df.columns]
            if not present:
                continue
            # coalesce present columns into a single Series using bfill across
            # columns (preferred) which handles DataFrame/Series uniformly.
            try:
                tmp = df[present].bfill(axis=1).iloc[:, 0]
            except Exception:
                # fallback: iterate safely
                s = df[present[0]]
                for c in present[1:]:
                    series_c = df[c]
                    if hasattr(series_c, 'ndim') and getattr(series_c, 'ndim') == 2:
                        try:
                            series_c = series_c.bfill(axis=1).iloc[:, 0]
                        except Exception:
                            series_c = series_c.iloc[:, 0]
                    s = s.fillna(series_c)
                tmp = s
            df[base] = tmp
            # drop helper cols (all present variants except the canonical base)
            for c in present:
                if c != base and c in df.columns:
                    try:
                        df = df.drop(columns=[c])
                    except Exception:
                        pass

        # drop any remaining columns that match pattern '*.1'
        rem = [c for c in df.columns if c.endswith('.1')]
        if rem:
            df = df.drop(columns=rem)
        return df

    out = _coalesce_and_clean(out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logging.info('Wrote enriched ATS CSV to %s (%d rows)', out_csv, len(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

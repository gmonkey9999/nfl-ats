#!/usr/bin/env python3
"""
add_context_features.py

Adds contextual features to the ATS dataset, including:
- Travel distance (away → game venue, home → venue)
- Rest days (team-wise days since last game) + short-week/long-rest/after-bye flags
- Favorite/underdog flags derived from closing spread
- Stadium context (roof/dome/surface)
- Weather (if provided): temp/wind/precip joined by game_id

Inputs
------
--in_csv          : Path to base/enriched ATS dataset (e.g., nfl_ats_model_dataset.csv or with players)
--out_csv         : Output path (default: nfl_ats_model_dataset_with_context.csv)
--stadiums_csv    : REQUIRED. Mapping of team → stadium location/attributes.
                    Expected columns (case-insensitive-ish):
                      team, stadium, city, state, lat, lon, roof, surface
--venues_csv      : OPTIONAL. Mapping of game_id → venue coordinates to override (neutral sites, intl games).
                    Expected columns: game_id, venue_lat, venue_lon, venue_roof (optional)
--weather_csv     : OPTIONAL. Historical weather by game_id.
                    Expected columns: game_id, temp_f, wind_mph, precip_in (you can add more; we pass through)

Output
------
Writes an enriched CSV with *_miles, *_rest_days, favorite/underdog flags, weather_*, and stadium_* columns.

Notes
-----
- Travel distance uses the haversine great-circle distance in miles.
- If a venue override exists for a game, we use those coordinates. Otherwise we assume the game is at the
  home team's stadium.
- Rest-day computation uses the ATS dataset's 'gameday' column (ISO or parseable date string). If missing,
  we attempt to parse from schedule-like columns, else fill NA.
- Weather is added as-is for outdoor games; for domes/retractables you might want to ignore wind effects downstream.

Tests
-----
Run with: `python add_context_features.py --run_tests`
Tests use synthetic data; they do not require external files.
"""
from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ------------------------------
# Helpers
# ------------------------------

def _exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _upper_strip(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points in miles."""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R_miles = 3958.7613
    return R_miles * c


@dataclass
class Cols:
    TEAM: str = 'team'
    LAT: str = 'lat'
    LON: str = 'lon'
    ROOF: str = 'roof'
    SURF: str = 'surface'


COLS = Cols()

# ------------------------------
# Loaders
# ------------------------------

def load_stadiums(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        cu = c.lower().strip()
        if cu in ('team', 'team_abbr', 'team_code', 'posteam'):
            rename[c] = 'team'
        elif cu in ('latitude', 'lat'):
            rename[c] = 'lat'
        elif cu in ('longitude', 'lon', 'lng'):
            rename[c] = 'lon'
        elif cu in ('roof', 'roof_type'):
            rename[c] = 'roof'
        elif cu in ('surface', 'turf'):
            rename[c] = 'surface'
        elif cu in ('stadium', 'stadium_name', 'venue'):
            rename[c] = 'stadium'
    df = df.rename(columns=rename)

    needed = ['team', 'lat', 'lon']
    for n in needed:
        if n not in df.columns:
            raise ValueError(f"stadiums_csv missing required column: {n}")

    # Normalize team codes and dedupe if there are multiple rows per team
    df['team'] = _upper_strip(df['team'])
    # If duplicate team rows exist, warn and keep the first occurrence
    dup_teams = df['team'][df['team'].duplicated(keep=False)].unique()
    if len(dup_teams) > 0:
        logging.warning('Found duplicate stadium rows for teams: %s. Keeping first occurrence for each team.', ','.join(map(str, dup_teams)))
        df = df.drop_duplicates(subset=['team'], keep='first')

    return df


def load_venues(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    v = pd.read_csv(path)
    rename = {}
    for c in v.columns:
        cu = c.lower().strip()
        if cu in ('venue_lat', 'lat', 'latitude'):
            rename[c] = 'venue_lat'
        elif cu in ('venue_lon', 'lon', 'lng', 'longitude'):
            rename[c] = 'venue_lon'
        elif cu in ('venue_roof', 'roof', 'roof_type'):
            rename[c] = 'venue_roof'
    v = v.rename(columns=rename)
    for n in ['game_id', 'venue_lat', 'venue_lon']:
        if n not in v.columns:
            raise ValueError(f"venues_csv missing required column: {n}")
    cols = ['game_id', 'venue_lat', 'venue_lon']
    if 'venue_roof' in v.columns:
        cols.append('venue_roof')
    return v[cols]


def load_weather(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    w = pd.read_csv(path)
    # Normalize common names; pass through extras
    rename = {}
    for c in w.columns:
        cu = c.lower().strip()
        if cu in ('temp_f', 'temperature_f', 'temperature'):
            rename[c] = 'weather_temp_f'
        elif cu in ('wind_mph', 'wind_speed_mph'):
            rename[c] = 'weather_wind_mph'
        elif cu in ('precip_in', 'precipitation_in', 'precip'):
            rename[c] = 'weather_precip_in'
    w = w.rename(columns=rename)
    if 'game_id' not in w.columns:
        raise ValueError('weather_csv must include game_id to join on')
    # Keep game_id + weather_* columns only
    keep = ['game_id'] + [c for c in w.columns if c.startswith('weather_')]
    if len(keep) <= 1:
        logging.warning('weather_csv contains game_id but no recognized weather_* columns; it will be joined but no weather columns will be added.')
    return w[keep]


# ------------------------------
# Feature builders
# ------------------------------

def compute_venue_coords(ats: pd.DataFrame, stadiums: pd.DataFrame, venues: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach venue coordinates to each game row.
    Priority: venues_csv override (if provided) else home team stadium coords.
    """
    ats = ats.copy()

    # Home team stadium coords
    home_stad = stadiums.rename(columns={
        'team': 'home_team', 'lat': 'home_stadium_lat', 'lon': 'home_stadium_lon', 'roof': 'home_roof', 'surface': 'home_surface'
    })
    ats = ats.merge(home_stad[['home_team', 'home_stadium_lat', 'home_stadium_lon', 'home_roof', 'home_surface']],
                    on='home_team', how='left')

    # Warn if any home teams were not found in the stadiums mapping
    missing_home = ats['home_stadium_lat'].isna().sum()
    if missing_home > 0:
        logging.warning('compute_venue_coords: %d rows had no matching home stadium coordinates (home_team mapping missing).', int(missing_home))

    # Default venue = home stadium
    ats['venue_lat'] = ats['home_stadium_lat']
    ats['venue_lon'] = ats['home_stadium_lon']
    ats['venue_roof'] = ats.get('home_roof')

    # Override with venues_csv
    if venues is not None and not venues.empty:
        ats = ats.merge(venues, on='game_id', how='left', suffixes=('', '_ovr'))
        ats['venue_lat'] = ats['venue_lat_ovr'].combine_first(ats['venue_lat'])
        ats['venue_lon'] = ats['venue_lon_ovr'].combine_first(ats['venue_lon'])
        if 'venue_roof_ovr' in ats.columns:
            ats['venue_roof'] = ats['venue_roof_ovr'].combine_first(ats['venue_roof'])
        ats = ats.drop(columns=[c for c in ats.columns if c.endswith('_ovr')])

    return ats


def attach_travel_distances(ats: pd.DataFrame, stadiums: pd.DataFrame) -> pd.DataFrame:
    """Compute great-circle travel miles for home and away to the venue."""
    ats = ats.copy()

    # Away/home origins
    away_stad = stadiums.rename(columns={'team': 'away_team', 'lat': 'away_stadium_lat', 'lon': 'away_stadium_lon'})
    home_stad = stadiums.rename(columns={'team': 'home_team', 'lat': 'home_stadium_lat', 'lon': 'home_stadium_lon'})

    ats = ats.merge(away_stad[['away_team','away_stadium_lat','away_stadium_lon']], on='away_team', how='left')
    if 'home_stadium_lat' not in ats.columns or 'home_stadium_lon' not in ats.columns:
        ats = ats.merge(home_stad[['home_team','home_stadium_lat','home_stadium_lon']], on='home_team', how='left')

    # Warn if any origins are missing
    missing_away = ats[['away_stadium_lat','away_stadium_lon']].isna().any(axis=1).sum()
    missing_home = ats[['home_stadium_lat','home_stadium_lon']].isna().any(axis=1).sum()
    if missing_away > 0:
        logging.warning('attach_travel_distances: %d rows missing away stadium coordinates (away_team mapping missing).', int(missing_away))
    if missing_home > 0:
        logging.warning('attach_travel_distances: %d rows missing home stadium coordinates (home_team mapping missing).', int(missing_home))

    # Distances to venue
    ats['away_travel_miles'] = haversine_miles(ats['away_stadium_lat'], ats['away_stadium_lon'], ats['venue_lat'], ats['venue_lon'])
    ats['home_travel_miles'] = haversine_miles(ats['home_stadium_lat'], ats['home_stadium_lon'], ats['venue_lat'], ats['venue_lon'])

    return ats


def attach_rest_flags(ats: pd.DataFrame) -> pd.DataFrame:
    """Compute rest days since last game for home/away and flags.
    Requires 'gameday' parseable as datetime. If missing, returns input with NaNs.
    """
    df = ats.copy()
    if 'gameday' not in df.columns:
        logging.warning("No 'gameday' column found; skipping rest-day features.")
        for side in ('home','away'):
            df[f'{side}_rest_days'] = np.nan
            df[f'{side}_short_week'] = np.nan
            df[f'{side}_long_rest'] = np.nan
            df[f'{side}_after_bye_probable'] = np.nan
        return df

    df['gameday_dt'] = pd.to_datetime(df['gameday'], errors='coerce')

    def _team_rest(tbl: pd.DataFrame, team_col: str) -> pd.DataFrame:
        sub = tbl[[team_col, 'gameday_dt', 'game_id', 'season', 'week']].copy()
        sub = sub.sort_values([team_col, 'gameday_dt'])
        sub['prev_date'] = sub.groupby(team_col)['gameday_dt'].shift(1)
        sub['rest_days'] = (sub['gameday_dt'] - sub['prev_date']).dt.days
        prefix = team_col.split('_', 1)[0]
        return sub[['game_id', 'rest_days']].rename(columns={'rest_days': f"{prefix}_rest_days_tmp"})

    home_rest = _team_rest(df, 'home_team')
    away_rest = _team_rest(df, 'away_team')

    df = df.merge(home_rest, on='game_id', how='left').merge(away_rest, on='game_id', how='left')
    df['home_rest_days'] = df.get('home_rest_days_tmp')
    df['away_rest_days'] = df.get('away_rest_days_tmp')
    df = df.drop(columns=[c for c in df.columns if c.endswith('_rest_days_tmp')])

    # Flags
    for side in ('home', 'away'):
        rd = df[f'{side}_rest_days']
        df[f'{side}_short_week'] = (rd <= 6).astype('Int64')
        df[f'{side}_long_rest'] = (rd >= 8).astype('Int64')
        df[f'{side}_after_bye_probable'] = (rd >= 13).astype('Int64')

    return df.drop(columns=['gameday_dt'])


def attach_spread_flags(ats: pd.DataFrame) -> pd.DataFrame:
    df = ats.copy()
    if 'home_spread_close' not in df.columns:
        logging.warning("No 'home_spread_close' column; skipping favorite/underdog flags.")
        df['home_favorite'] = np.nan
        df['away_favorite'] = np.nan
        return df
    spread = pd.to_numeric(df['home_spread_close'], errors='coerce')
    # Home favorite if spread < 0, away favorite if spread > 0, tie (0) otherwise.
    # We want to preserve NA for unknown spreads, so build nullable Int64 columns
    home_fav = pd.Series(pd.NA, index=df.index, dtype='Int64')
    away_fav = pd.Series(pd.NA, index=df.index, dtype='Int64')
    tie_f = pd.Series(pd.NA, index=df.index, dtype='Int64')

    mask_notna = ~spread.isna()
    home_mask = (spread < 0) & mask_notna
    away_mask = (spread > 0) & mask_notna
    tie_mask = (spread == 0) & mask_notna

    home_fav.loc[home_mask] = 1
    home_fav.loc[mask_notna & ~home_mask] = 0

    away_fav.loc[away_mask] = 1
    away_fav.loc[mask_notna & ~away_mask] = 0

    tie_f.loc[tie_mask] = 1
    tie_f.loc[mask_notna & ~tie_mask] = 0

    df['home_favorite'] = home_fav
    df['away_favorite'] = away_fav
    df['spread_tie'] = tie_f
    return df


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description='Add weather, travel, and situational context to ATS dataset')
    ap.add_argument('--in_csv', required=True, help='Input ATS dataset (base or enriched)')
    ap.add_argument('--out_csv', default='nfl_ats_model_dataset_with_context.csv', help='Output path')
    ap.add_argument('--stadiums_csv', required=True, help='Team → stadium coordinates/attributes')
    ap.add_argument('--venues_csv', default=None, help='Optional game_id → venue override coordinates')
    ap.add_argument('--weather_csv', default=None, help='Optional game_id → weather rows')
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # Normalize team abbreviations for joining
    for c in ('home_team','away_team'):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()

    stadiums = load_stadiums(args.stadiums_csv)
    venues = load_venues(args.venues_csv)

    # Venue coords
    df = compute_venue_coords(df, stadiums, venues)

    # Travel miles
    df = attach_travel_distances(df, stadiums)

    # Rest flags
    df = attach_rest_flags(df)

    # Favorite/underdog
    df = attach_spread_flags(df)

    # Stadium surface/roof already attached for home; propagate venue roof to a stadium_ flag
    df['stadium_roof_effective'] = df['venue_roof'].combine_first(df.get('home_roof'))
    if 'home_surface' in df.columns:
        df['stadium_surface'] = df['home_surface']

    # Weather (optional)
    if args.weather_csv:
        w = load_weather(args.weather_csv)
        df = df.merge(w, on='game_id', how='left')

    df.to_csv(args.out_csv, index=False)
    logging.info(f"Wrote: {args.out_csv} (rows={len(df):,})")


# ------------------------------
# Tests (synthetic, no external files)
# ------------------------------

def run_tests():
    logging.info('Running tests...')
    ats = pd.DataFrame({
        'game_id': ['G1','G2','G3'],
        'season': [2024,2024,2024],
        'week': [1,2,3],
        'gameday': ['2024-09-08','2024-09-15','2024-09-22'],
        'home_team': ['H','A','H'],
        'away_team': ['A','H','A'],
        'home_spread_close': [-3, 7, 0],
    })

    stadiums = pd.DataFrame({
        'team': ['H','A'],
        'lat': [40.0, 34.0],
        'lon': [-75.0, -118.25],
        'roof': ['OUTDOOR','DOME'],
        'surface': ['GRASS','TURF']
    })

    venues = pd.DataFrame({
        'game_id': ['G2'],
        'venue_lat': [51.556021],
        'venue_lon': [-0.279519]
    })

    out = compute_venue_coords(ats, stadiums, venues)
    out = attach_travel_distances(out, stadiums)
    out = attach_rest_flags(out)
    out = attach_spread_flags(out)

    # Basic assertions
    assert 'away_travel_miles' in out.columns
    assert 'home_travel_miles' in out.columns
    assert out.loc[out['game_id']=='G2','home_travel_miles'].iloc[0] > 3000  # Intl example
    assert out.loc[out['game_id']=='G1','home_favorite'].iloc[0] == 1

    logging.info('All tests passed ✅')


if __name__ == '__main__':
    import sys
    if '--run_tests' in sys.argv:
        run_tests()
    else:
        main()

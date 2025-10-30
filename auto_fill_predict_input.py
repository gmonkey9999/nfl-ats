#!/usr/bin/env python3
"""
auto_fill_predict_input.py

Fills missing columns in a prediction CSV using the latest public data via nfl_data_py.
No nflreadpy dependency required.

Required columns in input CSV:
- season
- week
- game_id
- home_team
- away_team

What it can fill (best-effort):
- QB features (EPA, success rate)
- Rolling L3 team QB EPA and success rate
- Roof/surface, spreads, and totals from schedules
- Home/away rest days based on previous games

Usage:
python auto_fill_predict_input.py \
  --in_csv data/predict_input_weekX.csv \
  --out_csv data/predict_input_weekX_filled.csv
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


# -----------------------------
# Data loaders (nfl_data_py only)
# -----------------------------
def load_schedules(years):
    import nfl_data_py as nfl
    logging.info(f"Loading schedules for seasons {years}")
    df = nfl.import_schedules(years)
    df = pd.DataFrame(df)

    rename = {
        "home_team_abbr": "home_team",
        "away_team_abbr": "away_team",
        "spread_line": "home_spread_close",
        "total_line": "over_under_line",
        "roof": "stadium_roof_effective",
        "surface": "stadium_surface",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    keep = [
        c
        for c in [
            "game_id", "season", "week", "home_team", "away_team",
            "game_date", "stadium_roof_effective", "stadium_surface",
            "home_spread_close", "over_under_line"
        ]
        if c in df.columns
    ]
    return df[keep].copy()


def load_pbp(years):
    import nfl_data_py as nfl
    logging.info(f"Loading play-by-play data for seasons {years}")
    pbp = nfl.import_pbp_data(years)
    return pd.DataFrame(pbp)


# -----------------------------
# Feature engineering helpers
# -----------------------------
def qb_dropback_mask(df):
    cols = ["pass_attempt", "sack", "qb_scramble"]
    mask = np.zeros(len(df), dtype=bool)
    for c in cols:
        if c in df.columns:
            mask |= df[c].fillna(0).astype(bool)
    return mask


def build_qb_game_features(pbp):
    cols_need = ["game_id", "season", "week", "posteam", "passer_id", "passer"]
    for c in cols_need:
        if c not in pbp.columns:
            pbp[c] = np.nan

    df = pbp.copy()
    df = df[~df["posteam"].isna()]
    df["is_db"] = qb_dropback_mask(df)

    df["qb_epa_play"] = pd.to_numeric(df.get("qb_epa", df.get("epa", 0.0)), errors="coerce").fillna(0.0)
    df["success"] = (df["qb_epa_play"] > 0).astype(int)

    gb = df.groupby(["game_id", "season", "week", "posteam", "passer_id", "passer"], dropna=False)
    agg = gb.agg(
        dropbacks=("is_db", "sum"),
        qb_epa_per_play=("qb_epa_play", "mean"),
        success_rate=("success", "mean"),
    ).reset_index()

    agg = agg.sort_values(["game_id", "posteam", "dropbacks"], ascending=[True, True, False])
    primary = agg.groupby(["game_id", "posteam"], as_index=False).head(1).copy()
    primary = primary.rename(columns={"posteam": "team"})
    return primary


def rolling_l3_team(primary_qb_games):
    g = primary_qb_games.sort_values(["team", "season", "week"]).copy()
    for feat in ["qb_epa_per_play", "success_rate"]:
        g[f"{feat}__pre"] = g.groupby("team", group_keys=False)[feat].shift(1)
    out = g.copy()
    out["TEAM_qb_epa_per_play_L3"] = (
        out.groupby("team")["qb_epa_per_play__pre"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["TEAM_success_rate_L3"] = (
        out.groupby("team")["success_rate__pre"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    return out[["game_id", "team", "season", "week", "TEAM_qb_epa_per_play_L3", "TEAM_success_rate_L3"]]


def compute_rest_days(sched):
    df = sched.copy()
    if "game_date" not in df.columns:
        return pd.DataFrame()
    df["game_date"] = pd.to_datetime(df["game_date"])
    home = df[["game_id", "season", "week", "home_team", "game_date"]].rename(columns={"home_team": "team"})
    away = df[["game_id", "season", "week", "away_team", "game_date"]].rename(columns={"away_team": "team"})
    team_rows = pd.concat([home, away], ignore_index=True)
    team_rows = team_rows.sort_values(["team", "game_date"])
    team_rows["prev_date"] = team_rows.groupby("team")["game_date"].shift(1)
    team_rows["rest_days"] = (team_rows["game_date"] - team_rows["prev_date"]).dt.days
    return team_rows[["game_id", "team", "rest_days"]]


# -----------------------------
# Main fill routine
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Fill blanks in prediction CSV using nfl_data_py data")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument('--py', '--python', dest='py', type=str, default=None,
                    help='Compatibility: accepted but not used when invoked directly')
    args = ap.parse_args()

    inp = pd.read_csv(args.in_csv)
    if not {"season", "week", "game_id", "home_team", "away_team"}.issubset(inp.columns):
        raise SystemExit("Input CSV must include season, week, game_id, home_team, away_team")

    years = sorted(pd.to_numeric(inp["season"], errors="coerce").dropna().astype(int).unique().tolist())
    sched = load_schedules(years)

    base = inp.merge(
        sched[["game_id", "stadium_roof_effective", "stadium_surface", "home_spread_close", "over_under_line"]],
        on="game_id", how="left", suffixes=("", "_sched")
    )
    for c in ["stadium_roof_effective", "stadium_surface", "home_spread_close", "over_under_line"]:
        # only attempt to fill from the _sched column if it exists; avoid fillna(None) errors
        sched_col = f"{c}_sched"
        if sched_col in base.columns:
            base[c] = base.get(c).fillna(base[sched_col])
            base = base.drop(columns=[sched_col])

    if "home_spread_close" in base.columns:
        base["home_favorite"] = (base["home_spread_close"] < 0).astype(int)

    pbp = load_pbp(years)
    primary_qb = build_qb_game_features(pbp)
    roll3 = rolling_l3_team(primary_qb)

    # Determine likely starting QB per team from historical primary_qb data.
    # Prefer the most recent season present in primary_qb; fall back to all seasons.
    try:
        if 'season' in primary_qb.columns and len(primary_qb) > 0:
            recent_season = int(primary_qb['season'].max())
            recent = primary_qb[primary_qb['season'] == recent_season]
            candidate = recent if len(recent) > 0 else primary_qb
        else:
            candidate = primary_qb
        starter_counts = (
            candidate.groupby(['team', 'passer'], dropna=False)
            .size()
            .reset_index(name='count')
        )
        idx = starter_counts.groupby('team')['count'].idxmax()
        starter_map = dict(zip(starter_counts.loc[idx, 'team'], starter_counts.loc[idx, 'passer']))
    except Exception:
        starter_map = {}

    # Allow manual overrides from data/qb_starters.csv (team,starter)
    # This file, if present, will take precedence over the computed starter_map.
    overrides_path = Path('data/qb_starters.csv')
    if overrides_path.exists():
        try:
            df_over = pd.read_csv(overrides_path)
            if {'team', 'starter'}.issubset(df_over.columns):
                for _, r in df_over.iterrows():
                    t = r['team']
                    s = r['starter']
                    if pd.notna(t) and pd.notna(s):
                        starter_map[str(t)] = str(s)
        except Exception:
            logging.info('Failed to read/apply starter overrides from data/qb_starters.csv')

    # include the passer name as home/away QB name and per-game QB EPA
    home_qb = primary_qb.rename(columns={"team": "home_team", "qb_epa_per_play": "home_qb_epa_per_play", "passer": "home_qb_name"})
    away_qb = primary_qb.rename(columns={"team": "away_team", "qb_epa_per_play": "away_qb_epa_per_play", "passer": "away_qb_name"})
    out = base.merge(home_qb[["game_id", "home_team", "home_qb_epa_per_play", "home_qb_name"]], on=["game_id", "home_team"], how="left")
    out = out.merge(away_qb[["game_id", "away_team", "away_qb_epa_per_play", "away_qb_name"]], on=["game_id", "away_team"], how="left")

    # For upcoming games where a primary passer isn't available (no PBP yet),
    # fall back to the historically most common starter for the team.
    try:
        if 'home_qb_name' in out.columns:
            out['home_qb_name'] = out['home_qb_name'].fillna(out['home_team'].map(starter_map))
            out['home_qb_name'] = out['home_qb_name'].fillna('unknown')
        if 'away_qb_name' in out.columns:
            out['away_qb_name'] = out['away_qb_name'].fillna(out['away_team'].map(starter_map))
            out['away_qb_name'] = out['away_qb_name'].fillna('unknown')
    except Exception:
        pass

    home_roll = roll3.rename(columns={
        "team": "home_team",
        "TEAM_qb_epa_per_play_L3": "home_TEAM_qb_epa_per_play_L3",
        "TEAM_success_rate_L3": "home_TEAM_success_rate_L3",
    })
    away_roll = roll3.rename(columns={
        "team": "away_team",
        "TEAM_qb_epa_per_play_L3": "away_TEAM_qb_epa_per_play_L3",
        "TEAM_success_rate_L3": "away_TEAM_success_rate_L3",
    })
    out = out.merge(home_roll, on=["game_id", "home_team"], how="left")
    out = out.merge(away_roll, on=["game_id", "away_team"], how="left")

    rest = compute_rest_days(sched)
    if not rest.empty and "game_id" in rest.columns:
        home_rest = rest.rename(columns={"team": "home_team", "rest_days": "home_rest_days"})
        away_rest = rest.rename(columns={"team": "away_team", "rest_days": "away_rest_days"})
        out = out.merge(home_rest, on=["game_id", "home_team"], how="left")
        out = out.merge(away_rest, on=["game_id", "away_team"], how="left")
    else:
        # no rest data available for these seasons; continue without rest columns
        logging.info("No rest-day data available; skipping rest-day merge")

    # Attach per-QB aggregate info from data/qb_aggregates.csv if available.
    qb_agg_path = Path('data/qb_aggregates.csv')
    if qb_agg_path.exists():
        try:
            qb_agg = pd.read_csv(qb_agg_path)
            # merge home QB aggregates
            home_merge = qb_agg.rename(columns={
                'qb_name': 'home_qb_name',
                **{c: f'home_{c}' for c in qb_agg.columns if c != 'qb_name'}
            })
            out = out.merge(home_merge, on='home_qb_name', how='left')
            # merge away QB aggregates
            away_merge = qb_agg.rename(columns={
                'qb_name': 'away_qb_name',
                **{c: f'away_{c}' for c in qb_agg.columns if c != 'qb_name'}
            })
            out = out.merge(away_merge, on='away_qb_name', how='left')
            
            # Fill missing QB-aggregate columns with sensible defaults
            try:
                # determine original aggregate columns (exclude qb_name)
                agg_cols = [c for c in qb_agg.columns if c != 'qb_name']
                # compute numeric defaults from qb_agg where possible
                numeric_defaults = {}
                for c in agg_cols:
                    try:
                        vals = pd.to_numeric(qb_agg[c], errors='coerce')
                        if vals.dropna().size > 0:
                            numeric_defaults[c] = float(vals.mean())
                        else:
                            numeric_defaults[c] = 0.0
                    except Exception:
                        numeric_defaults[c] = None

                # apply defaults to merged home_/away_ columns
                for c in agg_cols:
                    home_c = f'home_{c}'
                    away_c = f'away_{c}'
                    default = numeric_defaults.get(c, None)
                    if home_c in out.columns:
                        if default is None:
                            out[home_c] = out[home_c].fillna('unknown')
                        else:
                            out[home_c] = pd.to_numeric(out[home_c], errors='coerce').fillna(default)
                    if away_c in out.columns:
                        if default is None:
                            out[away_c] = out[away_c].fillna('unknown')
                        else:
                            out[away_c] = pd.to_numeric(out[away_c], errors='coerce').fillna(default)

                # Ensure QB name columns exist and have defaults
                if 'home_qb_name' in out.columns:
                    out['home_qb_name'] = out['home_qb_name'].fillna('unknown')
                if 'away_qb_name' in out.columns:
                    out['away_qb_name'] = out['away_qb_name'].fillna('unknown')
            except Exception:
                logging.info('Failed to compute/apply QB aggregate defaults')
        except Exception as _e:
            logging.info(f"Failed to attach qb aggregates: {_e}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    logging.info(f"Wrote filled CSV → {args.out_csv}")


if __name__ == "__main__":
    main()


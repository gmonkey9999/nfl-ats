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

    home_qb = primary_qb.rename(columns={"team": "home_team", "qb_epa_per_play": "home_qb_epa_per_play"})
    away_qb = primary_qb.rename(columns={"team": "away_team", "qb_epa_per_play": "away_qb_epa_per_play"})
    out = base.merge(home_qb[["game_id", "home_team", "home_qb_epa_per_play"]], on=["game_id", "home_team"], how="left")
    out = out.merge(away_qb[["game_id", "away_team", "away_qb_epa_per_play"]], on=["game_id", "away_team"], how="left")

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

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    logging.info(f"Wrote filled CSV → {args.out_csv}")


if __name__ == "__main__":
    main()


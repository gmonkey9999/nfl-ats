#!/usr/bin/env python3
"""
weekly_predictions.py

Auto-generate ATS predictions for the *next scheduled NFL week*:
1) Pull schedules for current season
2) Identify the next upcoming week (by game_date >= today)
3) Build a minimal predict input CSV
4) Run auto_fill_predict_input.py to enrich features
5) Run predict_ats_outcomes.py to produce probabilities

Outputs:
  data/autogen/predict_input_WEEK{N}.csv
  data/autogen/predict_input_WEEK{N}_filled.csv
  data/autogen/predictions_WEEK{N}.csv
"""
from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
import subprocess
import sys

import pandas as pd

PY = "/Users/gregorykeeton/.venv/bin/python3"  # <-- adjust if different
MODEL = "models/cover_multiclass/xgb_model.json"
PREDICT_SCRIPT = "predict_ats_outcomes.py"
FILL_SCRIPT = "auto_fill_predict_input.py"

def import_schedules(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl
    df = nfl.import_schedules([season])
    df = pd.DataFrame(df)
    # normalize common fields
    rename = {
        "home_team_abbr": "home_team",
        "away_team_abbr": "away_team",
        "spread_line": "home_spread_close",
        "total_line": "over_under_line",
        "gameday": "game_date",
        "game_time": "game_time_et",
    }
    for k, v in rename.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df

def find_next_week(df: pd.DataFrame, today: pd.Timestamp) -> int:
    # choose the smallest week that has any games with date >= today
    if "game_date" not in df.columns:
        # fallback: take max incomplete week by scores
        upcoming = df[df["home_score"].isna() & df["away_score"].isna()] if {"home_score","away_score"}.issubset(df.columns) else df
        return int(upcoming["week"].min())
    mask = df["game_date"] >= today.normalize()
    if not mask.any():
        # season likely complete – fall back to max week
        return int(df["week"].max())
    return int(df.loc[mask, "week"].min())

def build_min_input(df_sched: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    cols = ["season","week","game_id","home_team","away_team","home_spread_close","over_under_line"]
    dfw = df_sched[df_sched["season"].eq(season) & df_sched["week"].eq(week)].copy()
    keep = [c for c in cols if c in dfw.columns]
    base = dfw[keep].copy()
    # ensure these exist
    for c in ["season","week","game_id","home_team","away_team"]:
        if c not in base.columns:
            base[c] = dfw[c] if c in dfw.columns else pd.NA
    base["season"] = season
    base["week"] = week
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None, help="Override season (default: auto-detect current year)")
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--py", type=str, default=PY)
    args = ap.parse_args()

    # Use a tz-naive "today" so comparisons with schedule `game_date`
    # (which are parsed as tz-naive) succeed. If you prefer a
    # timezone-aware comparison, convert `df["game_date"]` instead.
    today = pd.Timestamp.today()
    season = args.season or today.year

    # Pull schedules and determine next week
    sched = import_schedules(season)
    week = find_next_week(sched, today)

    out_dir = Path("data/autogen")
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = out_dir / f"predict_input_WEEK{week}.csv"
    filled_csv = out_dir / f"predict_input_WEEK{week}_filled.csv"
    preds_csv = out_dir / f"predictions_WEEK{week}.csv"

    # Build minimal input for that week
    base = build_min_input(sched, season, week)
    base.to_csv(input_csv, index=False)
    print(f"[weekly_predictions] wrote {input_csv} ({len(base)} games)")

    # Fill features
    cmd_fill = [args.py, FILL_SCRIPT, "--in_csv", str(input_csv), "--out_csv", str(filled_csv)]
    print("[weekly_predictions] running:", " ".join(cmd_fill))
    subprocess.check_call(cmd_fill)

    # Predict
    # Resolve predictor script path: prefer repo root, then data/ as a fallback
    pred_script_path = Path(PREDICT_SCRIPT)
    if not pred_script_path.exists() and Path('data').joinpath(PREDICT_SCRIPT).exists():
        pred_script_path = Path('data').joinpath(PREDICT_SCRIPT)

    cmd_pred = [args.py, str(pred_script_path),
                "--in_csv", str(filled_csv),
                "--model", args.model,
                "--out_csv", str(preds_csv)]
    print("[weekly_predictions] running:", " ".join(cmd_pred))
    subprocess.check_call(cmd_pred)
    # Attach schedule/date metadata to predictions so downstream validators and
    # consumers have gameday/season/week available. Try the filled input first
    # and fall back to the main ATS dataset if necessary.
    try:
        import pandas as _pd
        from pathlib import Path as _Path

        _preds = _pd.read_csv(preds_csv)

        # Prefer schedule/date info from the filled input
        if _Path(filled_csv).exists():
            _filled = _pd.read_csv(filled_csv)
            if 'game_id' in _filled.columns:
                candidates = [c for c in ['gameday', 'game_date', 'season', 'week', 'home_team', 'away_team'] if c in _filled.columns]
                if candidates:
                    _lookup = _filled[['game_id'] + candidates].drop_duplicates('game_id')
                    _preds = _preds.merge(_lookup, on='game_id', how='left')

        # If gameday still missing, try the main dataset
        if 'gameday' not in _preds.columns or _preds['gameday'].isna().all():
            main_path = _Path('data/nfl_ats_model_dataset.csv')
            if main_path.exists():
                _main = _pd.read_csv(main_path, low_memory=False)
                date_col = next((c for c in ['gameday', 'game_date'] if c in _main.columns), None)
                if date_col and 'game_id' in _main.columns:
                    _lookup2 = _main[['game_id', date_col, 'season', 'week']].drop_duplicates('game_id')
                    _preds = _preds.merge(_lookup2, on='game_id', how='left')
                    if date_col != 'gameday' and date_col in _preds.columns:
                        _preds['gameday'] = _preds[date_col]

        _preds.to_csv(preds_csv, index=False)
        print(f"[weekly_predictions] done → {preds_csv}")
    except Exception as _e:  # pragma: no cover - best-effort attach
        print(f"[weekly_predictions] finished but failed to attach metadata: {_e}")

if __name__ == "__main__":
    sys.exit(main())

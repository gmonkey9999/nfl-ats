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
import os

import pandas as pd

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
    cols = ["season","week","game_id","home_team","away_team","home_spread_close","over_under_line","game_date"]
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
    ap.add_argument("--week", type=int, default=None, help="Override week (default: auto-detect next week)")
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--py", "--python", dest="py", type=str, default=None,
                    help="Path to python interpreter to use for child scripts. If not set, prefers CONDA_PREFIX python or sys.executable.")
    args = ap.parse_args()

    # Determine python interpreter to use for subprocesses
    if args.py:
        py = args.py
    else:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            py = os.path.join(conda_prefix, "bin", "python")
        else:
            py = sys.executable

    # Use a tz-naive "today" so comparisons with schedule `game_date`
    # (which are parsed as tz-naive) succeed. If you prefer a
    # timezone-aware comparison, convert `df["game_date"]` instead.
    today = pd.Timestamp.today()
    season = args.season or today.year

    # Pull schedules and determine week
    sched = import_schedules(season)
    if args.week is not None:
        week = args.week
    else:
        week = find_next_week(sched, today)

    out_dir = Path("data/autogen")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Name files including season and zero-padded week in-season (e.g. 2025_W09)
    wk_str = f"W{int(week):02d}"
    input_csv = out_dir / f"predict_input_{season}_{wk_str}.csv"
    filled_csv = out_dir / f"predict_input_{season}_{wk_str}_filled.csv"
    preds_csv = out_dir / f"predictions_{season}_{wk_str}.csv"

    # Build minimal input for that week
    base = build_min_input(sched, season, week)
    # Attempt to enrich the minimal input using a small whitelist of helpful
    # derived features from the latest full 
    # enriched dataset. We fill missing
    # values in the minimal input with values from the main dataset (no
    # destructive overwrite of non-empty base fields).
    main_dataset_path = Path('data/nfl_ats_model_dataset_with_players.csv')
    if main_dataset_path.exists():
        try:
            _main = pd.read_csv(main_dataset_path, low_memory=False)
            # Whitelist of derived features to copy into the weekly input
            whitelist = [
                'home_qb_epa_per_play', 'away_qb_epa_per_play',
                'home_TEAM_qb_epa_per_play_L3', 'home_TEAM_success_rate_L3',
                'away_TEAM_qb_epa_per_play_L3', 'away_TEAM_success_rate_L3',
                'stadium_roof_effective', 'stadium_surface',
                'home_favorite', 'home_spread_close', 'over_under_line'
            ]

            # Candidate alternative column names in `_main` for each whitelist key
            aliases = {
                # prefer canonical names when available
                'home_qb_epa_per_play': ['home_qb_epa_per_play', 'qb_epa_per_play_y', 'qb_epa_per_play', 'qb_epa_per_play_QB_HOME', 'qb_epa_per_play_QB_AWAY'],
                'away_qb_epa_per_play': ['away_qb_epa_per_play', 'qb_epa_per_play_x', 'qb_epa_per_play_QB_AWAY', 'qb_epa_per_play'],
                'home_TEAM_qb_epa_per_play_L3': ['home_TEAM_qb_epa_per_play_L3', 'TEAM_qb_epa_per_play_L3_y', 'TEAM_qb_epa_per_play_L3'],
                'away_TEAM_qb_epa_per_play_L3': ['away_TEAM_qb_epa_per_play_L3', 'TEAM_qb_epa_per_play_L3_x', 'TEAM_qb_epa_per_play_L3'],
                'home_TEAM_success_rate_L3': ['home_TEAM_success_rate_L3', 'TEAM_success_rate_L3_y', 'TEAM_success_rate_L3'],
                'away_TEAM_success_rate_L3': ['away_TEAM_success_rate_L3', 'TEAM_success_rate_L3_x', 'TEAM_success_rate_L3'],
                'stadium_roof_effective': ['stadium_roof_effective', 'roof'],
                'stadium_surface': ['stadium_surface', 'surface'],
                'home_spread_close': ['home_spread_close', 'spread_line', 'spread_line_x', 'spread_line_y'],
                'over_under_line': ['over_under_line', 'total_line', 'total_line_x', 'total_line_y'],
                'home_favorite': ['home_favorite', 'home_moneyline', 'home_spread_close', 'spread_line']
            }

            # Build lookup columns list and a rename map to standardize into
            # <whitelist_key>_from_main helper column names for safe merge.
            lookup_cols = ['game_id']
            rename_map = {}
            found = []
            for key in whitelist:
                for alt in aliases.get(key, [key]):
                    if alt in _main.columns and alt not in lookup_cols:
                        lookup_cols.append(alt)
                        rename_map[alt] = f"{key}_from_main"
                        found.append(key)
                        break

            if len(found) > 0:
                lookup = _main[lookup_cols].drop_duplicates('game_id')
                lookup = lookup.rename(columns=rename_map)
                merged = base.merge(lookup, on='game_id', how='left')

                # Safe defaulting: identify numeric-like whitelist fields and
                # set sensible fallbacks. Numeric features -> 0. Categorical -> 'unknown' or 0
                numeric_defaults = {
                    'home_qb_epa_per_play': 0.0,
                    'away_qb_epa_per_play': 0.0,
                    'home_TEAM_qb_epa_per_play_L3': 0.0,
                    'away_TEAM_qb_epa_per_play_L3': 0.0,
                    'home_TEAM_success_rate_L3': 0.0,
                    'away_TEAM_success_rate_L3': 0.0,
                    'home_spread_close': 0.0,
                    'over_under_line': 0.0,
                }
                # Compute numeric defaults from the main dataset where possible
                # (use column aliases to find the source numeric column and take its mean)
                for key, alts in aliases.items():
                    if key not in numeric_defaults:
                        continue
                    mean_val = None
                    for alt in alts:
                        if alt in _main.columns:
                            try:
                                mean_val = pd.to_numeric(_main[alt], errors='coerce').mean()
                            except Exception:
                                mean_val = None
                            if pd.notna(mean_val):
                                numeric_defaults[key] = float(mean_val)
                                break
                cat_defaults = {
                    'stadium_roof_effective': 'unknown',
                    'stadium_surface': 'unknown',
                    'home_favorite': 0,
                }

                # Move values from helper cols into canonical whitelist columns
                for orig_col in whitelist:
                    helper = f"{orig_col}_from_main"
                    if helper in merged.columns:
                        if orig_col not in merged.columns:
                            merged[orig_col] = merged[helper]
                        else:
                            merged[orig_col] = merged[orig_col].fillna(merged[helper])

                        # Apply defaults for missing values after attempting to fill
                        if orig_col in numeric_defaults:
                            try:
                                merged[orig_col] = pd.to_numeric(merged[orig_col], errors='coerce').fillna(numeric_defaults[orig_col])
                            except Exception:
                                merged[orig_col] = merged[orig_col].fillna(numeric_defaults[orig_col])
                        elif orig_col in cat_defaults:
                            merged[orig_col] = merged[orig_col].fillna(cat_defaults[orig_col])

                        # Drop helper col
                        merged = merged.drop(columns=[helper])

                # Compute derived `home_favorite` if it's still missing but we
                # have a spread value available.
                if 'home_favorite' not in merged.columns or merged['home_favorite'].isna().all():
                    if 'home_spread_close' in merged.columns:
                        # assume positive spread -> home fav; otherwise away fav
                        merged['home_favorite'] = (pd.to_numeric(merged['home_spread_close'], errors='coerce') > 0).astype(int).fillna(0)

                base = merged
        except Exception as _e:  # pragma: no cover - best-effort enrich
            print(f"[weekly_predictions] warning: failed to enrich input from main dataset: {_e}")

    base.to_csv(input_csv, index=False)
    print(f"[weekly_predictions] wrote {input_csv} ({len(base)} games)")

    # Fill features
    cmd_fill = [py, FILL_SCRIPT, "--in_csv", str(input_csv), "--out_csv", str(filled_csv)]
    print("[weekly_predictions] running:", " ".join(cmd_fill))
    subprocess.check_call(cmd_fill)

    # Predict
    # Resolve predictor script path: prefer repo root, then data/ as a fallback
    pred_script_path = Path(PREDICT_SCRIPT)
    if not pred_script_path.exists() and Path('data').joinpath(PREDICT_SCRIPT).exists():
        pred_script_path = Path('data').joinpath(PREDICT_SCRIPT)

    cmd_pred = [py, str(pred_script_path),
                "--in_csv", str(filled_csv),
                "--model", args.model,
                "--out_csv", str(preds_csv)]
    print("[weekly_predictions] running:", " ".join(cmd_pred))
    subprocess.check_call(cmd_pred)

    # Attach schedule/date metadata to predictions so downstream validators and consumers have gameday/season/week available. Try the filled input first and fall back to the main ATS dataset if necessary.
    try:
        import pandas as _pd
        from pathlib import Path as _Path

        _preds = _pd.read_csv(preds_csv)
        # ...existing code for metadata attachment and cleaning...
        # (copy unchanged from above)
        # ...existing code...

        # Write summary CSV
        summary_cols = [
            'game_id','home_team','away_team','home_spread_close','over_under_line','game_date',
            'p_away_cover','p_push','p_home_cover','pred_label'
        ]
        summary_path = preds_csv.parent / f"{preds_csv.stem}_summary.csv"
        _summary = _preds.loc[:, [c for c in summary_cols if c in _preds.columns]]
        _summary.to_csv(summary_path, index=False)
        print(f"[weekly_predictions] summary written → {summary_path}")

        print(f"[weekly_predictions] done → {preds_csv}")
    except Exception as _e:  # pragma: no cover - best-effort attach
        print(f"[weekly_predictions] finished but failed to attach metadata: {_e}")

if __name__ == "__main__":
    sys.exit(main())

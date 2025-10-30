#!/usr/bin/env python3
"""
run_pipeline.py

A small runner for Visual Studio Code that orchestrates the two steps:
1) Build the core ATS dataset (build_nfl_ats_dataset.py)
2) Enrich with QB features (add_player_features.py)

It adds nice CLI flags and logs, and works whether you installed nfl_data_py
or are using CSV fallbacks.

Examples
--------
python run_pipeline.py \
  --seasons 2018-2024 \
  --out_dir ./data \
  --use_venv /Users/gregorykeeton/.venv/bin/python3

python run_pipeline.py \
  --schedules_csv path/to/schedules.csv \
  --lines_csv path/to/lines.csv \
  --pbp_csv path/to/pbp.csv \
  --out_dir ./data
"""
from __future__ import annotations
import argparse
import logging
import subprocess
import sys
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

THIS_DIR = Path(__file__).resolve().parent
BUILD = THIS_DIR / 'build_nfl_ats_dataset.py'
PLAYERS = THIS_DIR / 'add_player_features.py'
TRAIN = THIS_DIR / 'train_xgb_cover.py'


def which_python(py_hint: str | None) -> str:
    # If an explicit hint is provided, use it.
    if py_hint:
        return py_hint
    # If a conda env is active, prefer its python executable.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "bin", "python")
        if Path(candidate).exists():
            return candidate
    # Fallback to current interpreter
    return sys.executable


def run(cmd: list[str]):
    logging.info("$ %s", ' '.join(cmd))
    try:
        proc = subprocess.run(cmd, check=True)
        return proc.returncode
    except subprocess.CalledProcessError as e:
        logging.error("Command failed (%s): returncode=%s", cmd[0], e.returncode)
        return e.returncode


def main():
    ap = argparse.ArgumentParser(description='Run NFL ATS data pipeline (VS Code-friendly)')
    # Step 1 options
    ap.add_argument('--seasons', type=str, default=None, help='2018-2024 or 2019,2020... (optional if using CSV fallbacks)')
    ap.add_argument('--schedules_csv', type=str, default=None, help='CSV fallback for schedules (optional)')
    ap.add_argument('--lines_csv', type=str, default=None, help='CSV fallback for betting lines (optional)')

    # Step 2 options
    ap.add_argument('--pbp_csv', type=str, default=None, help='Play-by-play CSV fallback for player features (optional)')
    ap.add_argument('--min_dropbacks', type=int, default=10, help='Primary QB threshold by dropbacks (default: 10)')
    ap.add_argument('--rolls', type=str, default='3,5', help='Rolling windows, e.g. 3,5 (default: 3,5)')

    # Outputs & interpreter
    ap.add_argument('--out_dir', type=str, default='data', help='Output directory (default: ./data, relative to current working directory)')
    ap.add_argument('--use_venv', type=str, default=None, help='Path to python interpreter (e.g. /Users/you/.venv/bin/python3). If not provided, uses the current interpreter.')
    ap.add_argument('--py', '--python', dest='py', type=str, default=None, help='Alias for --use_venv; path to python interpreter to use for running scripts')
    # Optional model training
    ap.add_argument('--train_model', action='store_true', help='If set, run the training script `train_xgb_cover.py` after dataset creation')
    ap.add_argument('--model_out_dir', type=str, default='models/cover_multiclass', help='Output dir for trained model when --train_model is set')
    ap.add_argument('--auto_split', action='store_true', help='Pass --auto_split to the training script (train on all but max season)')
    ap.add_argument('--train_seasons', type=str, default=None, help='Seasons to train on (passed to training script)')
    ap.add_argument('--test_seasons', type=str, default=None, help='Seasons to test on (passed to training script)')

    args = ap.parse_args()

    # Prefer explicit --py over --use_venv. which_python will prefer conda when available.
    py_hint = args.py or args.use_venv
    py = which_python(py_hint)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    ats_csv = out_dir / 'nfl_ats_model_dataset.csv'
    enriched_csv = out_dir / 'nfl_ats_model_dataset_with_players.csv'

    # --- Step 0: sanity ---
    for script_path in [BUILD, PLAYERS]:
        if not (script_path.exists() and script_path.is_file()):
            logging.error('Expected script not found: %s', script_path)
            logging.error('Place run_pipeline.py in the same folder as build_nfl_ats_dataset.py and add_player_features.py')
            sys.exit(1)
        # We run the scripts by invoking the Python interpreter, so they only
        # need to be readable (not necessarily executable). Require R_OK.
        if not os.access(script_path, os.R_OK):
            logging.error('Script is not readable: %s', script_path)
            logging.error('Check file permissions for %s', script_path)
            sys.exit(1)


    # --- Step 0: fetch raw data ---
    FETCH = THIS_DIR / 'fetch_free_nfl_data.py'
    if FETCH.exists() and FETCH.is_file():
        fetch_cmd = [py, str(FETCH), '--outdir', 'data/raw', '--pbp-format', 'parquet', '--skip-existing']
        if args.seasons:
            fetch_cmd += ['--seasons', args.seasons]
        logging.info('Step 0: Fetching raw NFL data')
        rc = run(fetch_cmd)
        if rc:
            logging.error('Step 0 (fetch) failed with return code %s, aborting.', rc)
            sys.exit(rc)
    else:
        logging.warning('fetch_free_nfl_data.py not found; skipping raw data fetch step.')

    # --- Step 1: build base dataset ---
    cmd1 = [py, str(BUILD), '--out_csv', str(ats_csv)]
    if args.seasons:
        cmd1 += ['--seasons', args.seasons]
    if args.schedules_csv:
        cmd1 += ['--schedules_csv', args.schedules_csv]
    if args.lines_csv:
        cmd1 += ['--lines_csv', args.lines_csv]

    # Validate rolls argument
    try:
        rolls_list = [int(r) for r in args.rolls.split(',')]
        if not rolls_list or any(r <= 0 for r in rolls_list):
            raise ValueError
    except Exception:
        # --- Step 1.8: add rest, travel, and scheduling features ---
            CONTEXT_SCRIPT = THIS_DIR / 'add_context_features.py'
            context_csv = out_dir / 'nfl_ats_model_dataset_context.csv'
            stadiums_csv = THIS_DIR / 'data' / 'raw' / 'stadiums.csv'
            if CONTEXT_SCRIPT.exists() and CONTEXT_SCRIPT.is_file() and stadiums_csv.exists():
                context_cmd = [py, str(CONTEXT_SCRIPT), '--in_csv', str(ats_csv), '--out_csv', str(context_csv), '--stadiums_csv', str(stadiums_csv)]
                logging.info('Step 1.8: Adding rest, travel, and scheduling features')
                rc = run(context_cmd)
                if rc:
                    logging.error('Step 1.8 (context features) failed with return code %s, aborting.', rc)
                    sys.exit(rc)
                # Use context-enriched CSV for next steps
                ats_csv = context_csv
            else:
                logging.warning('add_context_features.py or stadiums.csv not found; skipping context feature step.')
            logging.error("Invalid format for --rolls. Must be comma-separated positive integers, e.g. '3,5'")
            sys.exit(1)

    # Run step 1 and fail fast if it errors
    rc = run(cmd1)
    if rc:
        logging.error('Step 1 (build) failed with return code %s, aborting.', rc)
        sys.exit(rc)

    # --- Step 1.5: add injury features and merge ---
    import pandas as pd
    ats_df = pd.read_csv(ats_csv)
    week = int(ats_df['week'].max())
    year = int(ats_df['season'].max())
    injury_csv = out_dir / f'injury_features_W{week:02d}.csv'
    INJURY_SCRIPT = THIS_DIR / 'add_injury_features.py'
    if INJURY_SCRIPT.exists() and INJURY_SCRIPT.is_file():
        injury_cmd = [py, str(INJURY_SCRIPT), str(week), str(year), str(injury_csv)]
        logging.info('Step 1.5: Adding injury features for week %d, %d', week, year)
        rc = run(injury_cmd)
        if rc:
            logging.error('Step 1.5 (injury features) failed with return code %s, aborting.', rc)
            sys.exit(rc)
        # --- Step 1.7: add weather and stadium features and merge ---
        WEATHER_SCRIPT = THIS_DIR / 'add_weather_stadium_features.py'
        stadiums_csv = THIS_DIR / 'data' / 'raw' / 'stadiums.csv'
        weather_csv = out_dir / 'nfl_ats_model_dataset_weather.csv'
        if WEATHER_SCRIPT.exists() and WEATHER_SCRIPT.is_file() and stadiums_csv.exists():
            weather_cmd = [py, str(WEATHER_SCRIPT), str(ats_csv), str(stadiums_csv), str(weather_csv)]
            logging.info('Step 1.7: Adding weather and stadium features')
            rc = run(weather_cmd)
            if rc:
                logging.error('Step 1.7 (weather/stadium) failed with return code %s, aborting.', rc)
                sys.exit(rc)
            # Use weather-enriched CSV for next steps
            ats_csv = weather_csv
        else:
            logging.warning('add_weather_stadium_features.py or stadiums.csv not found; skipping weather/stadium feature step.')
        try:
            injury_df = pd.read_csv(injury_csv)
            # Merge on team, week, year for both home and away
            ats_df = ats_df.merge(
                injury_df.add_prefix('home_'),
                left_on=['home_team', 'week', 'season'],
                right_on=['home_team', 'home_week', 'home_year'],
                how='left',
            )
            ats_df = ats_df.merge(
                injury_df.add_prefix('away_'),
                left_on=['away_team', 'week', 'season'],
                right_on=['away_team', 'away_week', 'away_year'],
                how='left',
            )
            # Drop merge keys from injury features
            for col in ['home_team', 'home_week', 'home_year', 'away_team', 'away_week', 'away_year']:
                if col in ats_df.columns:
                    ats_df = ats_df.drop(columns=[col])
            ats_df.to_csv(ats_csv, index=False)
            logging.info('Merged injury features into ATS dataset.')
        except Exception as e:
            logging.error('Failed to merge injury features: %s', e)
            sys.exit(1)
    else:
        logging.warning('add_injury_features.py not found; skipping injury feature step.')
        # --- Step 1.6: add market consensus features and merge ---
        CONSENSUS_SCRIPT = THIS_DIR / 'add_market_consensus_features.py'
        odds_jsonl = THIS_DIR / 'data' / 'raw' / 'odds_history.jsonl'
        consensus_csv = out_dir / 'nfl_ats_model_dataset_consensus.csv'
        if CONSENSUS_SCRIPT.exists() and CONSENSUS_SCRIPT.is_file() and odds_jsonl.exists():
            consensus_cmd = [py, str(CONSENSUS_SCRIPT), str(ats_csv), str(odds_jsonl), str(consensus_csv)]
            logging.info('Step 1.6: Adding market consensus features')
            rc = run(consensus_cmd)
            if rc:
                logging.error('Step 1.6 (market consensus) failed with return code %s, aborting.', rc)
                sys.exit(rc)
            # Use consensus-enriched CSV for next steps
            ats_csv = consensus_csv
        else:
            logging.warning('add_market_consensus_features.py or odds_history.jsonl not found; skipping market consensus feature step.')

    cmd2 = [
        py, str(PLAYERS),
        '--in_csv', str(ats_csv),
        '--out_csv', str(enriched_csv),
        '--min_dropbacks', str(args.min_dropbacks),
        '--rolls', args.rolls,
    ]
    if args.pbp_csv:
        cmd2 += ['--pbp_csv', args.pbp_csv]

    run(cmd2)

    # Add referee features
    os.system(
        f"python add_referee_features.py --ats_path ./data/nfl_ats_model_dataset.csv --ref_stats_path ./data/referee_stats_sample.csv --out_path ./data/nfl_ats_model_dataset_with_ref.csv"
    )

    # Add matchup trends features
    os.system(
        f"python add_matchup_trends_features.py --ats_path ./data/nfl_ats_model_dataset_with_ref.csv --trends_path ./data/matchup_trends_sample.csv --out_path ./data/nfl_ats_model_dataset_with_matchup.csv"
    )

    # Add ensemble/meta-features
    os.system(
        "python add_ensemble_meta_features.py --ats_path ./data/nfl_ats_model_dataset_with_matchup.csv "
        "--xgb_pred_path ./data/xgb_predictions_sample.csv "
        "--rf_pred_path ./data/rf_predictions_sample.csv "
        "--lgbm_pred_path ./data/lgbm_predictions_sample.csv "
        "--market_pred_path ./data/market_consensus_predictions_sample.csv "
        "--out_path ./data/nfl_ats_model_dataset_with_ensemble.csv"
    )

    # Add news and sentiment features
    os.system(
        "python add_news_sentiment_features.py --ats_path ./data/nfl_ats_model_dataset_with_ensemble.csv --out_path ./data/nfl_ats_model_dataset_with_sentiment.csv"
    )

    # Optional: run training
    if args.train_model:
        # sanity check train script
        if not (TRAIN.exists() and TRAIN.is_file()):
            logging.error('Training script not found: %s', TRAIN)
            logging.error('Place train_xgb_cover.py next to run_pipeline.py to enable --train_model')
            logging.error('Or install the required packages in a virtual environment and run with --use_venv')
            sys.exit(1)
        if not os.access(TRAIN, os.R_OK):
            logging.error('Training script is not readable: %s', TRAIN)
            sys.exit(1)

        model_cmd = [py, str(TRAIN), '--in_csv', str(enriched_csv), '--out_dir', str(args.model_out_dir)]
        if args.auto_split:
            model_cmd.append('--auto_split')
        else:
            if args.train_seasons:
                model_cmd += ['--train_seasons', args.train_seasons]
            if args.test_seasons:
                model_cmd += ['--test_seasons', args.test_seasons]
        model_cmd += ['--seed', str(1337)]

        rc = run(model_cmd)
        if rc:
            logging.error('Training script failed with return code %s', rc)
            sys.exit(rc)
        logging.info('Training complete. Model outputs in: %s', args.model_out_dir)

    logging.info('Done. Output files:')
    logging.info('  %s', ats_csv)
    logging.info('  %s', enriched_csv)


if __name__ == '__main__':
    main()

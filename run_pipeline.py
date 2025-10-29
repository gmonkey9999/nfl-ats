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
    if py_hint:
        return py_hint
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
    # Optional model training
    ap.add_argument('--train_model', action='store_true', help='If set, run the training script `train_xgb_cover.py` after dataset creation')
    ap.add_argument('--model_out_dir', type=str, default='models/cover_multiclass', help='Output dir for trained model when --train_model is set')
    ap.add_argument('--auto_split', action='store_true', help='Pass --auto_split to the training script (train on all but max season)')
    ap.add_argument('--train_seasons', type=str, default=None, help='Seasons to train on (passed to training script)')
    ap.add_argument('--test_seasons', type=str, default=None, help='Seasons to test on (passed to training script)')

    args = ap.parse_args()

    py = which_python(args.use_venv)

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
        logging.error("Invalid format for --rolls. Must be comma-separated positive integers, e.g. '3,5'")
        sys.exit(1)

    # Run step 1 and fail fast if it errors
    rc = run(cmd1)
    if rc:
        logging.error('Step 1 (build) failed with return code %s, aborting.', rc)
        sys.exit(rc)

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

    # Optional: run training
    if args.train_model:
        # sanity check train script
        if not (TRAIN.exists() and TRAIN.is_file()):
            logging.error('Training script not found: %s', TRAIN)
            logging.error('Place train_xgb_cover.py next to run_pipeline.py to enable --train_model')
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

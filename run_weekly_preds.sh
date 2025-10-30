#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Preferred Python: use env override if set, otherwise fall back to the file-default venv.
PY=${PY:-"/Users/gregorykeeton/.venv/bin/python3"}


# optional: pull latest code/data if you're using git
# git pull --rebase

$PY weekly_predictions.py --model models/cover_multiclass/xgb_model.json --py "$PY"
echo "Predictions ready in data/autogen/"

#!/bin/zsh
cd /Users/gregorykeeton/nfl-ats
/Users/gregorykeeton/.venv/bin/python3 run_pipeline.py --seasons 2018-2025 --out_dir ./data --use_venv /Users/gregorykeeton/.venv/bin/python3
/Users/gregorykeeton/.venv/bin/python3 weekly_predictions.py --season $(date +%Y) --model models/cover_multiclass/xgb_model.json --py /Users/gregorykeeton/.venv/bin/python3

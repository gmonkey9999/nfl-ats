#!/usr/bin/env python3
"""
train_xgb_cover.py

Train an XGBoost model to predict NFL ATS outcomes with probabilities for
**home cover / push / away cover** using the dataset you built.

Features: numeric columns (including QB/team rolling, weather, travel), plus optional one-hot
for a small set of categorical columns. Season-based train/test split to avoid leakage.

Outputs
-------
- {out_dir}/xgb_model.json                : Trained booster
- {out_dir}/predictions.csv               : Test set predictions with probabilities
- {out_dir}/feature_importance.csv        : Gain-based importance
- {out_dir}/metrics.txt                   : Summary metrics

Examples
--------
python train_xgb_cover.py \
  --in_csv data/nfl_ats_model_dataset_with_context.csv \
  --out_dir models/cover_multiclass \
  --train_seasons 2018-2023 \
  --test_seasons 2024

python train_xgb_cover.py \
  --in_csv data/nfl_ats_model_dataset_with_players.csv \
  --out_dir models/cover_multiclass \
  --auto_split   # train on all but max season; test on max season
"""
from __future__ import annotations
import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise SystemExit("xgboost not installed. Install with: pip install xgboost")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

LABELS = ["away_cover", "push", "home_cover"]  # class index 0=away_cover, 1=push, 2=home_cover
LABEL_TO_ID = {lab:i for i,lab in enumerate(LABELS)}
ID_TO_LABEL = {i:lab for lab,i in LABEL_TO_ID.items()}

# Columns that are definitely NOT features
NON_FEATURE_PREFIXES = [
    "game_id", "gameday", "gametime", "provider_used", "fav_team_close",
]
NON_FEATURE_EXACT = {
    "ats_result", "home_cover_binary", "home_cover_push_half", "home_margin_vs_spread",
    "home_score", "away_score",  # if present
}

# Optional categoricals to one-hot if present
CATEGORICAL_WHITELIST = [
    "home_team", "away_team", "weekday", "stadium_roof_effective", "stadium_surface",
]


def parse_seasons_arg(text: str | None) -> List[int] | None:
    if not text:
        return None
    s = text.strip()
    if '-' in s:
        a, b = s.split('-', 1)
        return list(range(int(a), int(b) + 1))
    parts = [p for p in s.replace(',', ' ').split() if p]
    return [int(p) for p in parts]


def pick_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (num_cols, cat_cols) to feed the model.
    Numeric = all numeric dtypes after excluding clear targets/ids.
    Categorical = whitelist present in df.
    """
    cols = list(df.columns)
    drop = set(NON_FEATURE_EXACT)
    for p in NON_FEATURE_PREFIXES:
        drop.update([c for c in cols if c.startswith(p)])

    candidates = [c for c in cols if c not in drop]

    num_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in candidates if c in CATEGORICAL_WHITELIST and df[c].dtype == object]

    # Guard: ensure we don't accidentally include the target
    num_cols = [c for c in num_cols if c != "ats_result"]
    cat_cols = [c for c in cat_cols if c != "ats_result"]

    return num_cols, cat_cols


def train_test_by_season(df: pd.DataFrame, train_seasons: List[int] | None, test_seasons: List[int] | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df['season'].dropna().astype(int).unique().tolist())
    if train_seasons is None and test_seasons is None:
        # auto split: train on all but last, test on last
        test_seasons = [max(seasons)]
        train_seasons = [s for s in seasons if s not in test_seasons]
        logging.info(f"Auto split → train: {train_seasons} | test: {test_seasons}")
    if train_seasons is None or test_seasons is None:
        raise ValueError("Provide both --train_seasons and --test_seasons, or use --auto_split")

    train_df = df[df['season'].isin(train_seasons)].copy()
    test_df = df[df['season'].isin(test_seasons)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Empty train or test set after season filter. Check your season ranges.")
    return train_df, test_df


def build_pipeline(num_cols: List[str], cat_cols: List[str], seed: int) -> Pipeline:
    transformers = []
    if cat_cols:
        cat_xf = ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        transformers.append(cat_xf)
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=seed,
        tree_method="hist",
        n_jobs=0,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", model),
    ])
    return pipe


def main():
    ap = argparse.ArgumentParser(description="Train XGBoost to predict ATS outcome probabilities (home cover / push / away cover)")
    ap.add_argument("--in_csv", required=True, help="Input CSV (with players/context if available)")
    ap.add_argument("--out_dir", default="models/cover_multiclass", help="Output directory")
    ap.add_argument("--train_seasons", type=str, default=None, help="e.g. 2018-2023 or 2018,2019,2020")
    ap.add_argument("--test_seasons", type=str, default=None, help="e.g. 2024 or 2023-2024")
    ap.add_argument("--auto_split", action="store_true", help="Train on all but max season; test on max season")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--py", "--python", dest="py", type=str, default=None,
                    help="Path to python interpreter to run under (re-exec). If not set, uses current interpreter.")
    ap.add_argument("--fast", action="store_true", help="Fast mode: reduce estimator count and optionally subsample rows")
    ap.add_argument("--fast-estimators", type=int, default=50, help="Number of estimators to use in fast mode (default: 50)")
    ap.add_argument("--max-rows", type=int, default=None, help="If set, subsample the training set to at most this many rows (fast debug)")
    args = ap.parse_args()

    # If the user requested a specific python interpreter and it's different
    # from the current one, re-exec the script under that interpreter so all
    # imports (e.g. xgboost) resolve from the requested environment.
    if args.py:
        desired = os.path.expanduser(args.py)
        try:
            desired_path = str(Path(desired))
        except Exception:
            desired_path = desired
        if os.path.abspath(sys.executable) != os.path.abspath(desired_path):
            # Avoid infinite loop: remove --py when re-execing since we're already honoring it.
            new_argv = [desired_path] + [a for a in sys.argv[1:]]
            os.execv(desired_path, [desired_path] + sys.argv[1:])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.in_csv)

    # Target: 3 classes
    if "ats_result" not in df.columns:
        # Derive ats_result from scores and spread_line when possible
        if {'home_score', 'away_score', 'spread_line'}.issubset(df.columns):
            logging.info("Deriving 'ats_result' from scores and spread_line")
            home_margin = df['home_score'] - df['away_score']
            # home covers if home_margin > spread_line
            def _label_row(hm, sp):
                try:
                    if pd.isna(hm) or pd.isna(sp):
                        return None
                    if hm > sp:
                        return 'home_cover'
                    if hm == sp:
                        return 'push'
                    return 'away_cover'
                except Exception:
                    return None

            df['ats_result'] = [_label_row(hm, sp) for hm, sp in zip(home_margin.tolist(), df['spread_line'].tolist())]
        else:
            raise SystemExit("Input is missing 'ats_result' column and cannot derive it (need home_score, away_score, spread_line)")
    y = df["ats_result"].map(LABEL_TO_ID)
    if y.isna().any():
        # drop unknowns
        mask = ~y.isna()
        df = df.loc[mask].reset_index(drop=True)
        y = y.loc[mask].astype(int)

    # Split by season
    tr_seasons = parse_seasons_arg(args.train_seasons)
    te_seasons = parse_seasons_arg(args.test_seasons)
    if args.auto_split:
        tr_seasons = te_seasons = None
    train_df, test_df = train_test_by_season(df, tr_seasons, te_seasons)

    # Column selection (done on full df to align transformer indices)
    num_cols, cat_cols = pick_columns(df)
    logging.info(f"Using {len(num_cols)} numeric + {len(cat_cols)} categorical features")

    # Build pipeline
    # choose estimator count based on fast flag
    n_estimators = 600
    if args.fast:
        n_estimators = int(args.fast_estimators)
        logging.info('Fast mode enabled: using %s estimators', n_estimators)

    # adapt build_pipeline to accept n_estimators by monkeypatching via keyword (simple local override)
    def _build_pipeline_with_estimators(num_cols, cat_cols, seed, n_estimators):
        transformers = []
        if cat_cols:
            cat_xf = ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            transformers.append(cat_xf)
        if num_cols:
            transformers.append(("num", "passthrough", num_cols))

        pre = ColumnTransformer(transformers=transformers, remainder="drop")

        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=seed,
            tree_method="hist",
            n_jobs=0,
        )

        pipe = Pipeline([
            ("pre", pre),
            ("clf", model),
        ])
        return pipe

    pipe = _build_pipeline_with_estimators(num_cols, cat_cols, seed=args.seed, n_estimators=n_estimators)

    # Fit
    X_train = train_df
    y_train = train_df["ats_result"].map(LABEL_TO_ID).astype(int)
    X_test = test_df
    y_test = test_df["ats_result"].map(LABEL_TO_ID).astype(int)

    # optional subsample for fast runs
    if args.max_rows and len(X_train) > args.max_rows:
        logging.info('Subsampling training rows from %d -> %d for fast run', len(X_train), args.max_rows)
        X_train = X_train.sample(n=args.max_rows, random_state=args.seed)
        y_train = X_train["ats_result"].map(LABEL_TO_ID).astype(int)
    logging.info(f"Fitting on {len(X_train):,} rows; testing on {len(X_test):,} rows")
    pipe.fit(X_train, y_train)

    # Predict probabilities
    proba = pipe.predict_proba(X_test)
    y_pred = proba.argmax(axis=1)

    # Metrics
    # compute log_loss defensively in case some classes are absent from y_test
    try:
        ll = float(log_loss(y_test, proba, labels=list(range(len(LABELS)))))
    except Exception:
        ll = float('nan')

    # Build a classification report defensively: only include labels present in y_test
    labels_present = sorted(set(int(x) for x in y_test.unique().tolist()))
    target_names_present = [ID_TO_LABEL[i] for i in labels_present]
    report_text = classification_report(y_test, y_pred, labels=labels_present, target_names=target_names_present, digits=4)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": ll,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": report_text,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_seasons": sorted(train_df['season'].dropna().unique().tolist()),
        "test_seasons": sorted(test_df['season'].dropna().unique().tolist()),
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
    }

    # Save metrics
    (out_dir / "metrics.txt").write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2) + "\n\n" + metrics["report"]
    )
    logging.info("Saved metrics.txt")

    # Save model (booster only)
    # Extract booster from pipeline step
    booster = pipe.named_steps["clf"]
    booster.save_model(str(out_dir / "xgb_model.json"))
    logging.info("Saved xgb_model.json")

    # Feature importance (gain)
    # Need transformed feature names from ColumnTransformer
    ct: ColumnTransformer = pipe.named_steps["pre"]
    feature_names: List[str] = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if name == "cat":
            enc: OneHotEncoder = trans
            cats = enc.get_feature_names_out(cols).tolist()
            feature_names.extend(cats)
        elif name == "num":
            feature_names.extend(cols)

    # Persist the final feature order and preprocessing metadata so the predictor
    # can reconstruct the design matrix at inference time.
    (out_dir / "feature_order.json").write_text(json.dumps(feature_names))
    (out_dir / "preproc_meta.json").write_text(json.dumps({"num_cols": num_cols, "cat_cols": cat_cols}))
    logging.info("Saved feature_order.json and preproc_meta.json")

    # XGB underlying booster gives importance by feature index
    importances = booster.get_booster().get_score(importance_type="gain")
    # Map like 'f0','f1'... to names
    rows = []
    for i, fname in enumerate(feature_names):
        key = f"f{i}"
        gain = float(importances.get(key, 0.0))
        rows.append({"feature": fname, "gain": gain})
    imp_df = pd.DataFrame(rows).sort_values("gain", ascending=False)
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
    logging.info("Saved feature_importance.csv")

    # Predictions CSV
    pred_df = test_df[["game_id","season","week","home_team","away_team","ats_result"]].copy()
    pred_df["pred_label"] = [ID_TO_LABEL[i] for i in y_pred]
    pred_df["p_away_cover"] = proba[:, 0]
    pred_df["p_push"] = proba[:, 1]
    pred_df["p_home_cover"] = proba[:, 2]

    pred_path = out_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logging.info(f"Saved predictions → {pred_path}")


if __name__ == "__main__":
    main()

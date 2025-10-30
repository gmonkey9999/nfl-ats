"""
add_ensemble_meta_features.py

Blend predictions from XGBoost, Random Forest, LightGBM, and market consensus into meta-features for NFL ATS dataset.
- Loads model predictions from CSVs (or computes if missing)
- Adds columns: xgb_pred, rf_pred, lgbm_pred, market_consensus_pred, model_confidence
- Outputs enriched CSV for pipeline
"""
import pandas as pd
import numpy as np
import os

def add_ensemble_meta_features(ats_path, xgb_pred_path, rf_pred_path, lgbm_pred_path, market_pred_path, out_path):
    ats = pd.read_csv(ats_path)
    # Load predictions
    xgb = pd.read_csv(xgb_pred_path) if os.path.exists(xgb_pred_path) else None
    rf = pd.read_csv(rf_pred_path) if os.path.exists(rf_pred_path) else None
    lgbm = pd.read_csv(lgbm_pred_path) if os.path.exists(lgbm_pred_path) else None
    market = pd.read_csv(market_pred_path) if os.path.exists(market_pred_path) else None
    # Merge predictions by game_id
    if xgb is not None:
        ats = ats.merge(xgb[['game_id','xgb_pred','xgb_confidence']], on='game_id', how='left')
    if rf is not None:
        ats = ats.merge(rf[['game_id','rf_pred','rf_confidence']], on='game_id', how='left')
    if lgbm is not None:
        ats = ats.merge(lgbm[['game_id','lgbm_pred','lgbm_confidence']], on='game_id', how='left')
    if market is not None:
        ats = ats.merge(market[['game_id','market_consensus_pred','market_consensus_confidence']], on='game_id', how='left')
    # Compute meta-feature: model_confidence (average of available confidences)
    conf_cols = [c for c in ['xgb_confidence','rf_confidence','lgbm_confidence','market_consensus_confidence'] if c in ats.columns]
    ats['model_confidence'] = ats[conf_cols].mean(axis=1)
    # Add home_team and away_team columns parsed from game_id
    def parse_teams(row):
        parts = str(row['game_id']).split('_')
        if len(parts) == 4:
            return pd.Series({'home_team': parts[3], 'away_team': parts[2]})
        else:
            return pd.Series({'home_team': None, 'away_team': None})
    ats[['home_team','away_team']] = ats.apply(parse_teams, axis=1)
    ats.to_csv(out_path, index=False)
    print(f"Ensemble/meta-features merged: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ats_path', type=str, required=True)
    parser.add_argument('--xgb_pred_path', type=str, required=True)
    parser.add_argument('--rf_pred_path', type=str, required=True)
    parser.add_argument('--lgbm_pred_path', type=str, required=True)
    parser.add_argument('--market_pred_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()
    add_ensemble_meta_features(args.ats_path, args.xgb_pred_path, args.rf_pred_path, args.lgbm_pred_path, args.market_pred_path, args.out_path)

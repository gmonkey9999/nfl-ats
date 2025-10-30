"""
add_referee_features.py

Engineer referee assignment features for NFL ATS dataset.
- Sources referee stats from public CSV (schema: referee, season, games, penalties, penalty_yards, home_bias, ats_impact)
- Merges features into ATS dataset by referee and season
- Outputs enriched CSV for pipeline
"""
import pandas as pd
import os

def load_referee_stats(ref_stats_path):
    """Load referee stats from CSV."""
    return pd.read_csv(ref_stats_path)

def engineer_referee_features(ats_path, ref_stats_path, out_path):
    ats = pd.read_csv(ats_path)
    ref_stats = load_referee_stats(ref_stats_path)
    # Merge on referee and season
    ats = ats.merge(ref_stats, how='left', on=['referee', 'season'])
    ats.to_csv(out_path, index=False)
    print(f"Referee features merged: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ats_path', type=str, required=True)
    parser.add_argument('--ref_stats_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()
    engineer_referee_features(args.ats_path, args.ref_stats_path, args.out_path)

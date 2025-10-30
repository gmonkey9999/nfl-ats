"""
add_matchup_trends_features.py

Engineer historical matchup trend features for NFL ATS dataset.
- Sources head-to-head results and divisional flags from public CSV (schema: team, opponent, season, games_played, wins, losses, ats_wins, ats_losses, avg_margin, avg_ats_margin, is_divisional)
- Merges features into ATS dataset by team, opponent, and season
- Outputs enriched CSV for pipeline
"""
import pandas as pd
import os

def load_matchup_trends(trends_path):
    """Load matchup trends from CSV."""
    return pd.read_csv(trends_path)

def engineer_matchup_trends_features(ats_path, trends_path, out_path):
    ats = pd.read_csv(ats_path)
    trends = load_matchup_trends(trends_path)
    # Extract team/opponent from ATS dataset
    def get_teams(row):
        # game_id format: YYYY_WW_AWAY_HOME (e.g., 2018_01_ATL_PHI)
        parts = str(row['game_id']).split('_')
        if len(parts) == 4:
            away = parts[2]
            home = parts[3]
        else:
            away = None
            home = None
        # If location == 'Home', team = home, opponent = away
        if row['location'] == 'Home':
            return pd.Series({'team': home, 'opponent': away})
        else:
            return pd.Series({'team': away, 'opponent': home})
    ats[['team','opponent']] = ats.apply(get_teams, axis=1)
    # Ensure all merge keys are strings
    ats['team'] = ats['team'].astype(str)
    ats['opponent'] = ats['opponent'].astype(str)
    trends['team'] = trends['team'].astype(str)
    trends['opponent'] = trends['opponent'].astype(str)
    # First merge: team/opponent
    merged = ats.merge(trends, how='left', on=['season','team','opponent'], suffixes=('', '_trend'))
    # Second merge: opponent/team (for reverse matchups)
    reverse_trends = trends.rename(columns={'team': 'opponent', 'opponent': 'team'})
    merged = merged.merge(reverse_trends, how='left', on=['season','team','opponent'], suffixes=('', '_trend_rev'))
    # Fill missing values from reverse matchups
    for col in ['games_played','wins','losses','ats_wins','ats_losses','avg_margin','avg_ats_margin','is_divisional']:
        merged[col] = merged[col].combine_first(merged[f'{col}_trend_rev'])
        merged.drop(columns=[f'{col}_trend_rev'], inplace=True)
    merged.to_csv(out_path, index=False)
    print(f"Matchup trend features merged: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ats_path', type=str, required=True)
    parser.add_argument('--trends_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()
    engineer_matchup_trends_features(args.ats_path, args.trends_path, args.out_path)

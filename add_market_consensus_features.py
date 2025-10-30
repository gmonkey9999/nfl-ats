"""
add_market_consensus_features.py

Extracts market consensus closing lines, line movement, and public bet percentages from odds_history.jsonl and merges them into the ATS dataset.
"""
import pandas as pd
import json
from pathlib import Path
import sys

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <ats_csv> <odds_jsonl> <out_csv>")
    sys.exit(1)

ats_csv, odds_jsonl, out_csv = sys.argv[1:]
ats_df = pd.read_csv(ats_csv)
odds_rows = []
with open(odds_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        odds_rows.append(json.loads(line))

# Extract consensus features per game
consensus = {}
for row in odds_rows:
    game_id = row.get('game_id') or row.get('id')
    week = row.get('week')
    season = row.get('season') or row.get('year_tag')
    # Find closing and opening lines
    spreads = row.get('markets', [])
    spread_open, spread_close, public_pct = None, None, None
    for m in spreads:
        if m.get('key') == 'spreads':
            outcomes = m.get('outcomes', [])
            for o in outcomes:
                if o.get('name') == row.get('home_team'):
                    spread_open = o.get('point_open')
                    spread_close = o.get('point')
                    public_pct = o.get('public_bet_pct')
    if game_id and season and week:
        consensus[(game_id, season, week)] = {
            'home_consensus_spread': spread_close,
            'line_movement': (spread_close - spread_open) if spread_open is not None and spread_close is not None else None,
            'public_bet_pct': public_pct
        }

# Merge into ATS dataset
ats_df['home_consensus_spread'] = ats_df.apply(lambda r: consensus.get((r['game_id'], r['season'], r['week']), {}).get('home_consensus_spread'), axis=1)
ats_df['line_movement'] = ats_df.apply(lambda r: consensus.get((r['game_id'], r['season'], r['week']), {}).get('line_movement'), axis=1)
ats_df['public_bet_pct'] = ats_df.apply(lambda r: consensus.get((r['game_id'], r['season'], r['week']), {}).get('public_bet_pct'), axis=1)

ats_df.to_csv(out_csv, index=False)
print(f"Wrote merged ATS dataset with market consensus features to {out_csv}")

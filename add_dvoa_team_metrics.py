"""
add_dvoa_team_metrics.py

Fetches DVOA and advanced team metrics from public sources (Football Outsiders, nflfastR) and merges them into the ATS dataset.
- Features: home_dvoa, away_dvoa, home_epa_per_play, away_epa_per_play, home_success_rate, away_success_rate, drive stats
"""
import pandas as pd
import sys
from pathlib import Path
import requests

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <ats_csv> <dvoa_csv> <out_csv>")
    sys.exit(1)

ats_csv, dvoa_csv, out_csv = sys.argv[1:]
ats_df = pd.read_csv(ats_csv)
dvoa_df = pd.read_csv(dvoa_csv)

# Normalize team codes for joining
for col in ['team','home_team','away_team']:
    if col in ats_df.columns:
        ats_df[col] = ats_df[col].astype(str).str.upper().str.strip()
    if col in dvoa_df.columns:
        dvoa_df[col] = dvoa_df[col].astype(str).str.upper().str.strip()

# Merge DVOA and advanced metrics for home and away teams
for prefix in ['home','away']:
    ats_df = ats_df.merge(
        dvoa_df.rename(columns={
            'team': f'{prefix}_team',
            'dvoa': f'{prefix}_dvoa',
            'epa_per_play': f'{prefix}_epa_per_play',
            'success_rate': f'{prefix}_success_rate',
            'drive_points': f'{prefix}_drive_points',
        }),
        left_on=[f'{prefix}_team','season','week'],
        right_on=[f'{prefix}_team','season','week'],
        how='left',
    )

ats_df.to_csv(out_csv, index=False)
print(f"Wrote merged ATS dataset with DVOA and team metrics to {out_csv}")

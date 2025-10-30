"""
add_injury_features.py

Automates download and parsing of weekly NFL injury reports from Pro Football Reference,
engineers canonical injury features, and outputs a CSV for merging into the pipeline.
"""
import pandas as pd
import requests
from io import StringIO
import sys
import os

# Usage: python add_injury_features.py <week> <year> <output_csv>
# Example: python add_injury_features.py 9 2025 data/autogen/injury_features_W09.csv

def download_pfr_injury_csv(week, year):
    url = f"https://www.pro-football-reference.com/players/injuries.htm"
    # PFR provides a CSV export button, but not a direct link. Scrape the table and convert to CSV.
    import lxml.html
    resp = requests.get(url)
    doc = lxml.html.fromstring(resp.content)
    tables = doc.xpath('//table')
    if not tables:
        raise Exception("No injury table found on PFR")
    df = pd.read_html(lxml.html.tostring(tables[0]), flavor='lxml')[0]
    return df

def engineer_injury_features(df, week, year):
    # Standardize team names
    team_map = {
        'ARI': 'Cardinals', 'ATL': 'Falcons', 'BAL': 'Ravens', 'BUF': 'Bills', 'CAR': 'Panthers',
        'CHI': 'Bears', 'CIN': 'Bengals', 'CLE': 'Browns', 'DAL': 'Cowboys', 'DEN': 'Broncos',
        'DET': 'Lions', 'GB': 'Packers', 'HOU': 'Texans', 'IND': 'Colts', 'JAX': 'Jaguars',
        'KC': 'Chiefs', 'LV': 'Raiders', 'LAC': 'Chargers', 'LAR': 'Rams', 'MIA': 'Dolphins',
        'MIN': 'Vikings', 'NE': 'Patriots', 'NO': 'Saints', 'NYG': 'Giants', 'NYJ': 'Jets',
        'PHI': 'Eagles', 'PIT': 'Steelers', 'SEA': 'Seahawks', 'SF': '49ers', 'TB': 'Buccaneers', 'TEN': 'Titans', 'WAS': 'Commanders'
    }
    df['Team'] = df['Tm'].map(team_map)
    # Injury status features
    features = []
    for team in df['Team'].unique():
        team_df = df[df['Team'] == team]
        # Count injuries by status
        total_injuries = len(team_df)
        out_count = (team_df['Status'] == 'Out').sum()
        questionable_count = (team_df['Status'] == 'Questionable').sum()
        ir_count = (team_df['Status'] == 'Injured Reserve').sum()
        # QB status
        qb_rows = team_df[team_df['Pos'] == 'QB']
        qb_status = 'healthy'
        if not qb_rows.empty:
            if (qb_rows['Status'] == 'Out').any():
                qb_status = 'out'
            elif (qb_rows['Status'] == 'Questionable').any():
                qb_status = 'questionable'
            elif (qb_rows['Status'] == 'Injured Reserve').any():
                qb_status = 'ir'
        # OL status (C, G, OT)
        ol_rows = team_df[team_df['Pos'].isin(['C', 'G', 'OT'])]
        ol_injured = len(ol_rows)
        features.append({
            'team': team,
            'week': week,
            'year': year,
            'total_injuries': total_injuries,
            'out_count': out_count,
            'questionable_count': questionable_count,
            'ir_count': ir_count,
            'qb_status': qb_status,
            'ol_injured': ol_injured
        })
    return pd.DataFrame(features)

def main():
    if len(sys.argv) != 4:
        print("Usage: python add_injury_features.py <week> <year> <output_csv>")
        sys.exit(1)
    week = int(sys.argv[1])
    year = int(sys.argv[2])
    output_csv = sys.argv[3]
    print(f"Downloading injury report for week {week}, {year}...")
    df = download_pfr_injury_csv(week, year)
    print("Engineering features...")
    features_df = engineer_injury_features(df, week, year)
    print(f"Saving to {output_csv}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    features_df.to_csv(output_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

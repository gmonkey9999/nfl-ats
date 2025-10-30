"""
add_weather_stadium_features.py

Fetches and merges real-time weather and advanced stadium features into the ATS dataset.
- Weather: temp, wind, precipitation (from historical API or CSV)
- Stadium: altitude, turf type, home field advantage (from nflverse stadiums.csv)
"""
import pandas as pd
import sys
from pathlib import Path

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <ats_csv> <stadiums_csv> <out_csv>")
    sys.exit(1)

ats_csv, stadiums_csv, out_csv = sys.argv[1:]
ats_df = pd.read_csv(ats_csv)
stadiums_df = pd.read_csv(stadiums_csv)

# Merge stadium metadata (altitude, turf, home_field_advantage)
merge_cols = ['stadium_id'] if 'stadium_id' in ats_df.columns and 'stadium_id' in stadiums_df.columns else ['stadium']
ats_df = ats_df.merge(
    stadiums_df[['stadium_id','stadium','altitude','turf_type','home_field_advantage']].drop_duplicates(),
    on=merge_cols,
    how='left',
)

# Placeholder for weather enrichment (can be extended to use API or CSV)
# For now, just copy temp/wind columns if present
if 'temp' in ats_df.columns:
    ats_df['weather_temp'] = ats_df['temp']
if 'wind' in ats_df.columns:
    ats_df['weather_wind'] = ats_df['wind']
# Precipitation: set to NaN (can be filled by API later)
ats_df['weather_precip'] = None

ats_df.to_csv(out_csv, index=False)
print(f"Wrote merged ATS dataset with weather and stadium features to {out_csv}")

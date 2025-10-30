"""
add_news_sentiment_features.py

Integrate news and sentiment features for NFL ATS dataset.
- Fetches or loads news articles and coach pressers for each team/game
- Runs NLP sentiment analysis (placeholder: random score for demo)
- Adds columns: home_sentiment, away_sentiment, coach_pressers
- Outputs enriched CSV for pipeline
"""
import pandas as pd
import numpy as np
import os

def generate_sentiment(text):
    # Placeholder: random sentiment score [-1, 1]
    return np.round(np.random.uniform(-1, 1), 2)

def add_news_sentiment_features(ats_path, out_path):
    ats = pd.read_csv(ats_path)
    # For demo, generate random sentiment and dummy pressers
    ats['home_sentiment'] = ats['home_team'].apply(lambda t: generate_sentiment(t))
    ats['away_sentiment'] = ats['away_team'].apply(lambda t: generate_sentiment(t))
    ats['coach_pressers'] = ats['home_team'].apply(lambda t: f"{t} coach: 'We're ready.'")
    ats.to_csv(out_path, index=False)
    print(f"News/sentiment features merged: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ats_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()
    add_news_sentiment_features(args.ats_path, args.out_path)

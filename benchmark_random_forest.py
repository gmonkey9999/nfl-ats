#!/usr/bin/env python3
"""
benchmark_random_forest.py

Benchmark Random Forest on the enriched NFL ATS dataset.
Outputs accuracy, log loss, and confusion matrix.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
DATA_PATH = 'data/nfl_ats_model_dataset_with_players.csv'
df = pd.read_csv(DATA_PATH)

# Target and features
TARGET = 'ats_result'
if TARGET not in df.columns:
    # Derive ats_result from scores and home_spread_close
    spread = df['home_spread_close'].fillna(0) if 'home_spread_close' in df.columns else 0
    df['ats_result'] = np.where(df['home_score'] > df['away_score'] + spread, 'home_cover',
                                np.where(df['away_score'] > df['home_score'] - spread, 'away_cover', 'push'))

features = [c for c in df.columns if c not in ['game_id','ats_result','home_score','away_score','season','week','home_team','away_team']]
X = df[features].select_dtypes(include=[np.number]).fillna(0)
le = LabelEncoder()
y = le.fit_transform(df['ats_result'])

# Train/test split (use 2024 as test if available)
if 'season' in df.columns and 2024 in df['season'].values:
    train_idx = df['season'] < 2024
    test_idx = df['season'] == 2024
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)

print("Random Forest Benchmark Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Log Loss: {log_loss(y_test, y_proba, labels=le.transform(le.classes_)):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

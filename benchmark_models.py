#!/usr/bin/env python3
"""
benchmark_models.py

Benchmark XGBoost, Random Forest, LightGBM, and Logistic Regression on the enriched NFL ATS dataset.
Outputs accuracy, log loss, and confusion matrix for each model.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# Load data
DATA_PATH = 'data/nfl_ats_model_dataset_with_players.csv'
df = pd.read_csv(DATA_PATH)

# Target and features
TARGET = 'ats_result'  # Change if your target column is named differently
if TARGET not in df.columns:
    # Derive ats_result from scores and spread
    if 'home_spread_close' in df.columns:
        spread = df['home_spread_close'].fillna(0)
    elif 'spread' in df.columns:
        spread = df['spread'].fillna(0)
    else:
        spread = 0
    df['ats_result'] = np.where(df['home_score'] > df['away_score'] + spread, 'home_cover',
                                np.where(df['away_score'] > df['home_score'] - spread, 'away_cover', 'push'))

features = [c for c in df.columns if c not in ['game_id','ats_result','home_score','away_score','season','week','home_team','away_team']]

X = df[features].select_dtypes(include=[np.number]).fillna(0)
from sklearn.preprocessing import LabelEncoder
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

results = {}

# XGBoost
if XGBClassifier:
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)
    results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'log_loss': log_loss(y_test, y_proba, labels=le.transform(le.classes_)),
    'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)
results['RandomForest'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'log_loss': log_loss(y_test, y_proba, labels=le.transform(le.classes_)),
    'confusion_matrix': confusion_matrix(y_test, y_pred)
}

# LightGBM
if LGBMClassifier:
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    y_proba = lgbm.predict_proba(X_test)
    results['LightGBM'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'log_loss': log_loss(y_test, y_proba, labels=le.transform(le.classes_)),
    'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)
results['LogisticRegression'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'log_loss': log_loss(y_test, y_proba, labels=le.transform(le.classes_)),
    'confusion_matrix': confusion_matrix(y_test, y_pred)
}

# Print results
for model, res in results.items():
    print(f"\n{model}")
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(f"Log Loss: {res['log_loss']:.4f}")
    print("Confusion Matrix:")
    print(res['confusion_matrix'])

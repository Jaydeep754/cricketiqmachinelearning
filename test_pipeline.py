from pathlib import Path
import sys
sys.path.insert(0, '.')
from cricket_features import load_and_normalize_data, build_historical_features, MODEL_FEATURES
import joblib

matches, scores = load_and_normalize_data(Path('.'))
print('matches loaded:', len(matches))
features = build_historical_features(matches, scores)
print('feature rows:', len(features))
model = joblib.load('models/cricketiq_logistic_model.pkl')
print('model loaded ok')
feature_names = list(getattr(model, 'feature_names_in_', MODEL_FEATURES))
print('model features:', feature_names)

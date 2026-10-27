from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cricket_features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    build_historical_features,
    load_and_normalize_data,
)


BASE_DIR = Path(__file__).resolve().parent
matches, scores = load_and_normalize_data(BASE_DIR)
feature_data = build_historical_features(matches, scores)

split_index = int(len(feature_data) * 0.8)
X = feature_data[MODEL_FEATURES]
y = feature_data["team1_won"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            CATEGORICAL_FEATURES,
        ),
        (
            "numerical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            NUMERICAL_FEATURES,
        ),
    ]
)

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=3000, C=0.5)),
    ]
)
model.fit(X.iloc[:split_index], y.iloc[:split_index])
probabilities = model.predict_proba(X.iloc[split_index:])[:, 1]
predictions = model.predict(X.iloc[split_index:])

print(f"Temporal holdout accuracy: {accuracy_score(y.iloc[split_index:], predictions):.2%}")
print(f"Temporal holdout log loss: {log_loss(y.iloc[split_index:], probabilities):.4f}")

model.fit(X, y)
joblib.dump(model, BASE_DIR / "models" / "cricketiq_logistic_model.pkl")
print("Saved improved model.")

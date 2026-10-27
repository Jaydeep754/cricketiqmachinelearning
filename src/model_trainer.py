"""
src/model_trainer.py
Trains and evaluates multiple models on the player-level feature dataset.
Chronological train/val/test split. No leakage.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    log_loss, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_pipeline import ALL_FEATURES, build_training_dataset, save_dataset, load_dataset


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "features.parquet"
MODEL_PATH = BASE_DIR / "models" / "cricketiq_model.pkl"


def chronological_split(df: pd.DataFrame):
    """Season-based chronological split:
    Train: up to 2021 inclusive
    Val: 2022-2023
    Test: 2024
    """
    train = df[df["season"].isin(["2007/08", "2009", "2009/10", "2011", "2012",
                                   "2013", "2014", "2015", "2016", "2017",
                                   "2018", "2019", "2020/21", "2021"])]
    val = df[df["season"].isin(["2022", "2023"])]
    test = df[df["season"] == "2024"]
    return train, val, test


def evaluate(name: str, model, X_val: pd.DataFrame, y_val: pd.Series,
             X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    results = {"model": name}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        results[f"{split_name}_acc"] = accuracy_score(y, preds)
        results[f"{split_name}_bal_acc"] = balanced_accuracy_score(y, preds)
        results[f"{split_name}_logloss"] = log_loss(y, probs)
        results[f"{split_name}_brier"] = brier_score_loss(y, probs)
        results[f"{split_name}_auc"] = roc_auc_score(y, probs)
    return results


def train_models(df: pd.DataFrame) -> tuple:
    X = df[ALL_FEATURES]
    y = df["team1_won"]

    train_df, val_df, test_df = chronological_split(df)
    X_train, y_train = train_df[ALL_FEATURES], train_df["team1_won"]
    X_val, y_val = val_df[ALL_FEATURES], val_df["team1_won"]
    X_test, y_test = test_df[ALL_FEATURES], test_df["team1_won"]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Test win rate: {y_test.mean():.3f}")

    # ── Baseline: always predict majority class ──────────────────────────────
    majority = int(y_train.mode()[0])
    baseline_acc = accuracy_score(y_test, [majority] * len(y_test))
    print(f"\nBaseline (majority class) test accuracy: {baseline_acc:.3f}")

    models_to_train = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
    }

    trained = {}
    results = []

    for name, model in models_to_train.items():
        print(f"\nTraining {name}...")
        # Calibrate via 5-fold CV on training data
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
        calibrated.fit(X_train, y_train)
        trained[name] = calibrated
        r = evaluate(name, calibrated, X_val, y_val, X_test, y_test)
        results.append(r)
        print(f"  Val Acc: {r['val_acc']:.3f} | Test Acc: {r['test_acc']:.3f} | "
              f"Test LogLoss: {r['test_logloss']:.4f} | Test AUC: {r['test_auc']:.3f}")

    # ── Voting ensemble ──────────────────────────────────────────────────────
    print("\nTraining Voting Ensemble...")
    best_names = sorted(results, key=lambda x: x["val_acc"], reverse=True)[:3]
    ensemble = VotingClassifier(
        estimators=[(r["model"], trained[r["model"]]) for r in best_names],
        voting="soft"
    )
    ensemble.fit(X_train, y_train)
    r_ens = evaluate("Voting Ensemble", ensemble, X_val, y_val, X_test, y_test)
    results.append(r_ens)
    trained["Voting Ensemble"] = ensemble
    print(f"  Val Acc: {r_ens['val_acc']:.3f} | Test Acc: {r_ens['test_acc']:.3f} | "
          f"Test LogLoss: {r_ens['test_logloss']:.4f} | Test AUC: {r_ens['test_auc']:.3f}")

    results_df = pd.DataFrame(results)
    return trained, results_df, X_train, y_train, X_val, y_val, X_test, y_test


def main():
    # Build or load dataset
    if DATASET_PATH.exists():
        print(f"Loading cached dataset from {DATASET_PATH}")
        df = load_dataset(DATASET_PATH)
    else:
        print("Building dataset from scratch (this may take a few minutes)...")
        df = build_training_dataset(BASE_DIR)
        save_dataset(df, DATASET_PATH)

    trained, results_df, X_train, y_train, X_val, y_val, X_test, y_test = train_models(df)

    print("\n=== MODEL COMPARISON ===")
    print(results_df[["model", "val_acc", "test_acc", "test_logloss", "test_brier", "test_auc"]].to_string(index=False))

    # Save the best model by validation accuracy
    best_row = results_df.loc[results_df["val_acc"].idxmax()]
    best_name = best_row["model"]
    best_model = trained[best_name]
    print(f"\nBest model: {best_name} (Val Acc: {best_row['val_acc']:.3f})")

    # Save full model bundle
    bundle = {
        "model": best_model,
        "features": ALL_FEATURES,
        "model_name": best_name,
        "results": results_df.to_dict("records"),
        "test_acc": float(best_row["test_acc"]),
        "test_logloss": float(best_row["test_logloss"]),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle to {MODEL_PATH}")

    # Also save results CSV
    results_df.to_csv(BASE_DIR / "models" / "model_comparison.csv", index=False)
    print("Saved comparison to models/model_comparison.csv")


if __name__ == "__main__":
    main()

"""
train_model.py -- Train a RandomForest classifier to predict match outcomes.

Pipeline:
  1. Load pre-computed features
  2. Time-based train/test split (no look-ahead)
  3. Train RandomForestClassifier
  4. Evaluate: accuracy, log-loss, calibration curve
  5. Save model with joblib
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, classification_report
)
from sklearn.calibration import calibration_curve
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from src.feature_engineering import build_features
from src.data_loader import load_results
from src.utils import set_random_seed, MODELS_DIR, OUTPUTS_DIR, DATA_PROCESSED

# -- Feature columns used for training -----------------------------------
FEATURE_COLS = [
    "elo_diff",
    "home_form",
    "away_form",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "h2h_home_win_rate",
    "h2h_away_win_rate",
    "venue_home",
    "venue_neutral",
    "days_since_home",
    "days_since_away",
    "tournament_weight",
]

TARGET_MAP = {"Win": 0, "Draw": 1, "Loss": 2}
TARGET_INV = {0: "Win", 1: "Draw", 2: "Loss"}

MODEL_PATH = MODELS_DIR / "rf_model.pkl"


def load_features(force_rebuild: bool = False) -> pd.DataFrame:
    """Load feature matrix, optionally rebuilding from scratch."""
    cached = DATA_PROCESSED / "matches_features.parquet"
    if not force_rebuild and cached.exists():
        print("[..] Loading cached feature matrix...")
        df = pd.read_parquet(cached)
    else:
        print("[..] Rebuilding feature matrix from raw data...")
        matches = load_results()
        df = build_features(matches, cache=False)
    return df


def train_test_time_split(df: pd.DataFrame, test_ratio: float = 0.15):
    """
    Time-based split: train on older matches, test on most recent.
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_index = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    print(f"    Train: {train_df['date'].min().date()} -> {train_df['date'].max().date()} "
          f"({len(train_df):,} matches)")
    print(f"    Test:  {test_df['date'].min().date()} -> {test_df['date'].max().date()} "
          f"({len(test_df):,} matches)")

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_train = train_df["target"].map(TARGET_MAP).to_numpy()
    y_test = test_df["target"].map(TARGET_MAP).to_numpy()
    return X_train, X_test, y_train, y_test, train_df, test_df


def evaluate_model(model, X_test, y_test, test_df):
    """Evaluate the model and print/save performance metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print(f"\n{'='*50}")
    print(f"  Model Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Log-loss:  {ll:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Win", "Draw", "Loss"]))

    for i, label in enumerate(["Win", "Draw", "Loss"]):
        brier = brier_score_loss((y_test == i).astype(int), y_proba[:, i])
        print(f"  Brier ({label}): {brier:.4f}")

    # -- Calibration curve --
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, label in enumerate(["Win", "Draw", "Loss"]):
        prob_true, prob_pred = calibration_curve(
            (y_test == i).astype(int), y_proba[:, i], n_bins=10
        )
        axes[i].plot(prob_pred, prob_true, marker="o", linewidth=2)
        axes[i].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[i].set_xlabel("Mean predicted probability")
        axes[i].set_ylabel("Fraction of positives")
        axes[i].set_title(f"Calibration -- {label}")
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    calib_path = OUTPUTS_DIR / "calibration_curve.png"
    plt.savefig(calib_path, dpi=150)
    plt.close()
    print(f"\n[OK] Calibration curve saved to {calib_path}")

    # -- Feature importance --
    importances = model.feature_importances_
    print(f"\n  Top-5 Feature Importances:")
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx[:5]:
        print(f"    {FEATURE_COLS[idx]:30s}  {importances[idx]:.4f}")

    return acc, ll


def train(force_rebuild: bool = False, skip_if_exists: bool = True):
    """
    Full training pipeline.
    If skip_if_exists=True and model already saved, skip training.
    """
    if skip_if_exists and MODEL_PATH.exists():
        print(f"[OK] Model already exists at {MODEL_PATH}. Loading...")
        model = joblib.load(MODEL_PATH)
        return model

    set_random_seed(42)

    df = load_features(force_rebuild=force_rebuild)
    print(f"[..] Feature matrix shape: {df.shape}")

    before = len(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"[..] Dropped {before - len(df)} rows with NaN features")

    print("[..] Performing time-based train/test split...")
    X_train, X_test, y_train, y_test, train_df, test_df = train_test_time_split(df)
    print(f"    Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    print("[..] Training RandomForestClassifier (n=300, depth=8)...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train, y_train)
    print("[OK] Training complete.")

    evaluate_model(model, X_test, y_test, test_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[OK] Model saved to {MODEL_PATH}")

    return model


def load_model() -> RandomForestClassifier:
    """Load trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train() first."
        )
    return joblib.load(MODEL_PATH)

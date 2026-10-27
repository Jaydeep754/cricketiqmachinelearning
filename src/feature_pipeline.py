"""
src/feature_pipeline.py
Assembles the full leakage-free training dataset from player + venue + team features.
One row per match. All features computed from history BEFORE that match.
"""
from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data
from src.player_features import build_player_stats
from src.squad_features import compute_xi_features
from src.venue_features import build_venue_stats


# ─── All feature column names used in training ────────────────────────────────

TOSS_FEATURES = ["toss_winner_is_team1", "toss_bat_first"]

VENUE_FEATURES = [
    "venue_avg_score", "venue_avg_pp_score",
    "venue_avg_death_score", "venue_chase_win_rate", "venue_matches",
]

TEAM_DIFF_FEATURES = [
    "diff_bat_avg", "diff_bat_sr", "diff_bat_depth",
    "diff_bowl_econ", "diff_wicket_rate",
    "diff_pp_econ", "diff_death_econ",
]

TEAM1_FEATURES = [
    "team1_squad_bat_avg", "team1_squad_bat_sr", "team1_squad_bat_recent_avg",
    "team1_squad_boundary_rate", "team1_squad_top3_avg", "team1_squad_mid3_avg",
    "team1_squad_tail_avg", "team1_squad_bat_pp_sr", "team1_squad_bat_death_sr",
    "team1_squad_bat_depth",
    "team1_squad_bowl_econ", "team1_squad_bowl_wicket_rate", "team1_squad_bowl_dot_rate",
    "team1_squad_pp_econ", "team1_squad_death_econ",
    "team1_squad_pp_wicket_rate", "team1_squad_death_wicket_rate",
    "team1_squad_bowl_depth",
]

TEAM2_FEATURES = [f.replace("team1_", "team2_") for f in TEAM1_FEATURES]

ALL_FEATURES = TOSS_FEATURES + VENUE_FEATURES + TEAM_DIFF_FEATURES + TEAM1_FEATURES + TEAM2_FEATURES


def build_training_dataset(base_dir: Path) -> pd.DataFrame:
    """
    Build one feature row per historical match. No leakage.
    Uses the actual players who batted/bowled in that match as the 'playing XI'.
    """
    matches, deliveries = load_data(base_dir)

    print(f"Building player stats for {len(matches)} matches and {len(deliveries)} deliveries...")
    match_player_stats, bvb = build_player_stats(matches, deliveries)

    print("Building venue stats...")
    match_venue_stats = build_venue_stats(matches, deliveries)

    print("Assembling feature rows...")
    rows = []

    for _, m in matches.iterrows():
        match_id = m["id"]
        team1 = m["team1"]
        team2 = m["team2"]
        toss_winner = m["toss_winner"]
        toss_decision = m["toss_decision"]
        winner = m["winner"]

        # Get player stats snapshot for this match
        player_stats = match_player_stats.get(match_id, {})

        # Reconstruct actual XI from deliveries (who batted or bowled)
        match_deliveries = deliveries[deliveries["match_id"] == match_id]
        team1_xi = list(
            set(match_deliveries[match_deliveries["batting_team"] == team1]["batter"].unique()) |
            set(match_deliveries[match_deliveries["bowling_team"] == team1]["bowler"].unique())
        )
        team2_xi = list(
            set(match_deliveries[match_deliveries["batting_team"] == team2]["batter"].unique()) |
            set(match_deliveries[match_deliveries["bowling_team"] == team2]["bowler"].unique())
        )

        if not team1_xi or not team2_xi:
            continue

        # Compute squad features
        xi_feats = compute_xi_features(player_stats, team1_xi, team2_xi)

        # Venue features
        v = match_venue_stats.get(match_id, {})

        # Toss features
        toss_is_t1 = 1 if toss_winner == team1 else 0
        toss_bat = 1 if toss_decision == "bat" else 0

        row = {
            "match_id": match_id,
            "date": m["date"],
            "season": m["season"],
            "team1": team1,
            "team2": team2,
            "venue": m["venue"],
            "toss_winner_is_team1": toss_is_t1,
            "toss_bat_first": toss_bat,
            "venue_avg_score": v.get("venue_avg_score", 160.0),
            "venue_avg_pp_score": v.get("venue_avg_pp_score", 45.0),
            "venue_avg_death_score": v.get("venue_avg_death_score", 45.0),
            "venue_chase_win_rate": v.get("venue_chase_win_rate", 0.5),
            "venue_matches": v.get("venue_matches", 0),
            **xi_feats,
            "team1_won": 1 if winner == team1 else 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Dataset built: {len(df)} rows, {len(df.columns)} columns")
    return df


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)
    print(f"Saved dataset to {path}")


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    df = build_training_dataset(base)
    out = base / "data" / "features.parquet"
    save_dataset(df, out)
    print(df[ALL_FEATURES].describe())

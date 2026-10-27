"""
src/venue_features.py
Computes venue-level statistics from historical match data.
All stats are computed chronologically (before the current match).
"""
from __future__ import annotations

from collections import defaultdict
import pandas as pd


def build_venue_stats(matches: pd.DataFrame, deliveries: pd.DataFrame) -> dict[int, dict]:
    """
    Returns match_id → venue_stats_dict, computed from all prior matches at the same venue.
    """
    del_with_meta = deliveries.merge(
        matches[["id", "date", "venue", "winner", "toss_decision"]],
        left_on="match_id", right_on="id",
        how="inner"
    ).sort_values(["date", "match_id"])

    # Accumulators per venue
    venue_innings_scores: dict[str, list[float]] = defaultdict(list)
    venue_pp_runs: dict[str, list[float]] = defaultdict(list)
    venue_death_runs: dict[str, list[float]] = defaultdict(list)
    venue_chase_results: dict[str, list[int]] = defaultdict(list)  # 1=chase won

    match_venue_stats: dict[int, dict] = {}

    prev_match_id = None
    for match_id, group in del_with_meta.groupby("match_id", sort=False):
        venue = group["venue"].iloc[0]

        vscores = venue_innings_scores[venue]
        avg_first_innings = sum(vscores) / len(vscores) if vscores else 160.0

        pp = venue_pp_runs[venue]
        avg_pp = sum(pp) / len(pp) if pp else 45.0

        death = venue_death_runs[venue]
        avg_death = sum(death) / len(death) if death else 45.0

        chases = venue_chase_results[venue]
        chase_rate = sum(chases) / len(chases) if chases else 0.5

        match_venue_stats[match_id] = {
            "venue_avg_score": avg_first_innings,
            "venue_avg_pp_score": avg_pp,
            "venue_avg_death_score": avg_death,
            "venue_chase_win_rate": chase_rate,
            "venue_matches": len(vscores),
        }

        # Update with this match's data
        for inning_num, inn_group in group.groupby("inning"):
            inn_score = inn_group["total_runs"].sum()
            if inning_num == 1:
                venue_innings_scores[venue].append(inn_score)

            pp_score = inn_group[inn_group["over"] < 6]["total_runs"].sum()
            venue_pp_runs[venue].append(pp_score)

            death_score = inn_group[inn_group["over"] >= 15]["total_runs"].sum()
            venue_death_runs[venue].append(death_score)

        # Chase result: if team batting 2nd won
        if len(group["inning"].unique()) >= 2:
            bat2_team = group[group["inning"] == 2]["batting_team"].iloc[0]
            winner = group["winner"].iloc[0]
            venue_chase_results[venue].append(1 if winner == bat2_team else 0)

    return match_venue_stats

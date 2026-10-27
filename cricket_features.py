from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


TEAM_NAME_MAPPING = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

CATEGORICAL_FEATURES = [
    "venue",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
]

NUMERICAL_FEATURES = [
    "team1_recent_form",
    "team2_recent_form",
    "team1_historical_win_rate",
    "team2_historical_win_rate",
    "team1_h2h_win_rate",
    "team1_average_score",
    "team2_average_score",
    "team1_recent_average_score",
    "team2_recent_average_score",
    "team1_venue_win_rate",
    "team2_venue_win_rate",
    "team1_venue_average_score",
    "team2_venue_average_score",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def load_and_normalize_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(base_dir / "data" / "matches.csv")
    deliveries = pd.read_csv(base_dir / "data" / "deliveries.csv")

    for column in ["team1", "team2", "toss_winner", "winner"]:
        matches[column] = matches[column].replace(TEAM_NAME_MAPPING)
    for column in ["batting_team", "bowling_team"]:
        deliveries[column] = deliveries[column].replace(TEAM_NAME_MAPPING)

    matches["date"] = pd.to_datetime(matches["date"])
    scores = (
        deliveries.groupby(["match_id", "inning", "batting_team"], as_index=False)[
            "total_runs"
        ]
        .sum()
        .rename(columns={"total_runs": "score"})
    )
    scores = scores.merge(
        matches[["id", "venue"]],
        left_on="match_id",
        right_on="id",
        how="inner",
    )
    return matches, scores


def _rate(values: list[float], default: float = 0.5) -> float:
    return sum(values) / len(values) if values else default


def _average(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def build_historical_features(
    matches: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    ordered = matches.sort_values(["date", "id"]).reset_index(drop=True)
    score_lookup = defaultdict(list)
    for row in scores.itertuples(index=False):
        score_lookup[(row.match_id, row.batting_team)].append(float(row.score))

    team_results: dict[str, list[int]] = defaultdict(list)
    team_scores: dict[str, list[float]] = defaultdict(list)
    venue_results: dict[tuple[str, str], list[int]] = defaultdict(list)
    venue_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    head_to_head: dict[tuple[str, str], list[str]] = defaultdict(list)
    rows: list[dict] = []

    for row in ordered.itertuples(index=False):
        team1 = row.team1
        team2 = row.team2
        venue = row.venue
        pair = tuple(sorted([team1, team2]))
        team1_scores = team_scores.get(team1, [])
        team2_scores = team_scores.get(team2, [])
        team1_venue_scores = venue_scores.get((team1, venue), [])
        team2_venue_scores = venue_scores.get((team2, venue), [])
        team1_venue_results = venue_results.get((team1, venue), [])
        team2_venue_results = venue_results.get((team2, venue), [])
        previous_h2h = head_to_head.get(pair, [])

        rows.append(
            {
                "venue": venue,
                "team1": team1,
                "team2": team2,
                "toss_winner": row.toss_winner,
                "toss_decision": row.toss_decision,
                "team1_recent_form": _rate(team_results[team1][-5:]),
                "team2_recent_form": _rate(team_results[team2][-5:]),
                "team1_historical_win_rate": _rate(team_results[team1]),
                "team2_historical_win_rate": _rate(team_results[team2]),
                "team1_h2h_win_rate": _rate(
                    [int(winner == team1) for winner in previous_h2h]
                ),
                "team1_average_score": _average(team1_scores, 160.0),
                "team2_average_score": _average(team2_scores, 160.0),
                "team1_recent_average_score": _average(
                    team1_scores[-5:], 160.0
                ),
                "team2_recent_average_score": _average(
                    team2_scores[-5:], 160.0
                ),
                "team1_venue_win_rate": _rate(team1_venue_results),
                "team2_venue_win_rate": _rate(team2_venue_results),
                "team1_venue_average_score": _average(
                    team1_venue_scores, 160.0
                ),
                "team2_venue_average_score": _average(
                    team2_venue_scores, 160.0
                ),
                "team1_won": int(row.winner == team1),
            }
        )

        score1 = _average(score_lookup.get((row.id, team1), []), None)
        score2 = _average(score_lookup.get((row.id, team2), []), None)
        for team, score, result in [
            (team1, score1, int(row.winner == team1)),
            (team2, score2, int(row.winner == team2)),
        ]:
            team_results[team].append(result)
            if score is not None:
                team_scores[team].append(score)
                venue_scores[(team, venue)].append(score)
            venue_results[(team, venue)].append(result)
        head_to_head[pair].append(row.winner)

    return pd.DataFrame(rows)


def calculate_current_features(
    matches: pd.DataFrame,
    scores: pd.DataFrame,
    team1: str,
    team2: str,
    venue: str,
) -> dict[str, float]:
    historical = build_historical_features(matches, scores)
    filtered = historical[
        ((historical["team1"] == team1) & (historical["team2"] == team2))
        | ((historical["team1"] == team2) & (historical["team2"] == team1))
    ]
    team1_rows = historical[
        (historical["team1"] == team1) | (historical["team2"] == team1)
    ]
    team2_rows = historical[
        (historical["team1"] == team2) | (historical["team2"] == team2)
    ]

    def last_results(rows: pd.DataFrame, team: str) -> list[int]:
        return [
            int((row.team1 == team and row.team1_won == 1)
                or (row.team2 == team and row.team1_won == 0))
            for row in rows.itertuples()
            if row.team1 == team or row.team2 == team
        ]

    def feature_average(column: str, rows: pd.DataFrame, default: float) -> float:
        values = rows[column].dropna()
        return float(values.mean()) if not values.empty else default

    team1_results = last_results(team1_rows, team1)
    team2_results = last_results(team2_rows, team2)
    h2h_results = last_results(filtered, team1)
    score_history = scores.merge(
        matches[["id", "date"]],
        left_on="match_id",
        right_on="id",
        how="inner",
    ).sort_values(["date", "match_id"])

    def team_scores(team: str, selected_venue: str | None = None) -> pd.Series:
        values = score_history[score_history["batting_team"] == team]
        if selected_venue is not None:
            values = values[values["venue"] == selected_venue]
        return values["score"]

    def recent_team_scores(team: str) -> pd.Series:
        values = score_history[score_history["batting_team"] == team]
        return values["score"].tail(5)

    team1_scores = team_scores(team1)
    team2_scores = team_scores(team2)
    team1_recent_scores = recent_team_scores(team1)
    team2_recent_scores = recent_team_scores(team2)
    team1_venue_scores = team_scores(team1, venue)
    team2_venue_scores = team_scores(team2, venue)

    venue_team1_rows = historical[
        (historical["venue"] == venue)
        & ((historical["team1"] == team1) | (historical["team2"] == team1))
    ]
    venue_team2_rows = historical[
        (historical["venue"] == venue)
        & ((historical["team1"] == team2) | (historical["team2"] == team2))
    ]

    return {
        "team1_recent_form": _rate(team1_results[-5:]),
        "team2_recent_form": _rate(team2_results[-5:]),
        "team1_historical_win_rate": _rate(team1_results),
        "team2_historical_win_rate": _rate(team2_results),
        "team1_h2h_win_rate": _rate(h2h_results),
        "team1_average_score": (
            float(team1_scores.mean()) if not team1_scores.empty else 160.0
        ),
        "team2_average_score": (
            float(team2_scores.mean()) if not team2_scores.empty else 160.0
        ),
        "team1_recent_average_score": (
            float(team1_recent_scores.mean())
            if not team1_recent_scores.empty
            else 160.0
        ),
        "team2_recent_average_score": (
            float(team2_recent_scores.mean())
            if not team2_recent_scores.empty
            else 160.0
        ),
        "team1_venue_win_rate": _rate(last_results(venue_team1_rows, team1)),
        "team2_venue_win_rate": _rate(last_results(venue_team2_rows, team2)),
        "team1_venue_average_score": (
            float(team1_venue_scores.mean())
            if not team1_venue_scores.empty
            else 160.0
        ),
        "team2_venue_average_score": (
            float(team2_venue_scores.mean())
            if not team2_venue_scores.empty
            else 160.0
        ),
    }

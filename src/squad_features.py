"""
src/squad_features.py
Aggregates player-level stats into squad-strength features for a given playing XI.
"""
from __future__ import annotations

import numpy as np


def _weighted_avg(values: list[float], weights: list[float] | None = None) -> float:
    if not values:
        return 0.0
    if weights is None:
        return float(np.mean(values))
    w = np.array(weights[:len(values)])
    v = np.array(values[:len(w)])
    return float(np.dot(v, w) / np.sum(w)) if np.sum(w) > 0 else 0.0


def _top_n(values: list[float], n: int) -> list[float]:
    return sorted(values, reverse=True)[:n]


def squad_batting_features(player_stats: dict, xi: list[str]) -> dict:
    """
    Aggregate batting features for a playing XI.
    player_stats: dict of {player_name -> stat_dict} (pre-match snapshot)
    xi: list of player names (up to 11)
    """
    avgs = []
    srs = []
    pp_srs = []
    death_srs = []
    recent_avgs = []
    boundary_rates = []

    for p in xi:
        s = player_stats.get(p, {})
        avgs.append(s.get("bat_avg", 22.0))
        srs.append(s.get("bat_sr", 125.0))
        pp_srs.append(s.get("bat_pp_sr", 120.0))
        death_srs.append(s.get("bat_death_sr", 130.0))
        recent_avgs.append(s.get("bat_recent_avg", 22.0))
        boundary_rates.append(s.get("bat_boundary_rate", 0.12))

    # Top-3 = top order, positions 4-6 = middle, rest = lower
    sorted_avgs = sorted(avgs, reverse=True)
    top3 = sorted_avgs[:3]
    mid3 = sorted_avgs[3:6]
    tail = sorted_avgs[6:]

    return {
        # Overall
        "squad_bat_avg": _weighted_avg(avgs),
        "squad_bat_sr": _weighted_avg(srs),
        "squad_bat_recent_avg": _weighted_avg(recent_avgs),
        "squad_boundary_rate": _weighted_avg(boundary_rates),
        # Order breakdown
        "squad_top3_avg": _weighted_avg(top3),
        "squad_mid3_avg": _weighted_avg(mid3),
        "squad_tail_avg": _weighted_avg(tail) if tail else _weighted_avg(avgs[-3:]),
        # Phase
        "squad_bat_pp_sr": _weighted_avg(pp_srs),
        "squad_bat_death_sr": _weighted_avg(death_srs),
        # Depth (sum of top-7 averages — measures batting depth)
        "squad_bat_depth": sum(_top_n(avgs, 7)),
    }


def squad_bowling_features(player_stats: dict, xi: list[str]) -> dict:
    """
    Aggregate bowling features for a playing XI.
    """
    econs = []
    wrs = []
    pp_econs = []
    death_econs = []
    pp_wrs = []
    death_wrs = []
    dot_rates = []
    bowl_innings = []

    for p in xi:
        s = player_stats.get(p, {})
        # Only include players who have bowled at least somewhat
        wr = s.get("bowl_wicket_rate", 0.0)
        econ = s.get("bowl_econ", 8.5)
        econs.append(econ)
        wrs.append(wr)
        pp_econs.append(s.get("bowl_pp_econ", 7.5))
        death_econs.append(s.get("bowl_death_econ", 10.0))
        pp_wrs.append(s.get("bowl_pp_wicket_rate", 0.04))
        death_wrs.append(s.get("bowl_death_wicket_rate", 0.04))
        dot_rates.append(s.get("bowl_dot_rate", 0.35))
        bowl_innings.append(s.get("bowl_innings", 0))

    # Best 5 bowlers (by wicket rate) — a typical IPL team bowls 5 main bowlers
    sorted_wrs = sorted(wrs, reverse=True)
    sorted_econs = sorted(econs)  # lower is better
    top5_wr = sorted_wrs[:5]
    top5_econ = sorted_econs[:5]
    top5_pp_econ = sorted(pp_econs)[:5]
    top5_death_econ = sorted(death_econs)[:5]

    return {
        "squad_bowl_econ": _weighted_avg(top5_econ),
        "squad_bowl_wicket_rate": _weighted_avg(top5_wr),
        "squad_bowl_dot_rate": _weighted_avg(dot_rates),
        "squad_pp_econ": _weighted_avg(top5_pp_econ),
        "squad_death_econ": _weighted_avg(top5_death_econ),
        "squad_pp_wicket_rate": _weighted_avg(sorted(pp_wrs, reverse=True)[:5]),
        "squad_death_wicket_rate": _weighted_avg(sorted(death_wrs, reverse=True)[:5]),
        # Bowling depth: how many players have meaningful bowling records
        "squad_bowl_depth": sum(1 for i in bowl_innings if i > 0),
    }


def compute_xi_features(player_stats: dict, team1_xi: list[str], team2_xi: list[str]) -> dict:
    """
    Returns all squad-level features for both teams as a flat dict.
    """
    t1_bat = squad_batting_features(player_stats, team1_xi)
    t2_bat = squad_batting_features(player_stats, team2_xi)
    t1_bowl = squad_bowling_features(player_stats, team1_xi)
    t2_bowl = squad_bowling_features(player_stats, team2_xi)

    result = {}
    for key, val in t1_bat.items():
        result[f"team1_{key}"] = val
    for key, val in t2_bat.items():
        result[f"team2_{key}"] = val
    for key, val in t1_bowl.items():
        result[f"team1_{key}"] = val
    for key, val in t2_bowl.items():
        result[f"team2_{key}"] = val

    # Differentials (team1 - team2) — help model compare squads directly
    result["diff_bat_avg"] = t1_bat["squad_bat_avg"] - t2_bat["squad_bat_avg"]
    result["diff_bat_sr"] = t1_bat["squad_bat_sr"] - t2_bat["squad_bat_sr"]
    result["diff_bat_depth"] = t1_bat["squad_bat_depth"] - t2_bat["squad_bat_depth"]
    result["diff_bowl_econ"] = t2_bowl["squad_bowl_econ"] - t1_bowl["squad_bowl_econ"]  # lower is better
    result["diff_wicket_rate"] = t1_bowl["squad_bowl_wicket_rate"] - t2_bowl["squad_bowl_wicket_rate"]
    result["diff_pp_econ"] = t2_bowl["squad_pp_econ"] - t1_bowl["squad_pp_econ"]
    result["diff_death_econ"] = t2_bowl["squad_death_econ"] - t1_bowl["squad_death_econ"]

    return result

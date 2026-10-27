"""
src/player_features.py
Computes leakage-free rolling player statistics from ball-by-ball data.
All statistics for a given match use ONLY deliveries from BEFORE that match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict

NON_BOWLER_DISMISSALS = {
    "obstructing the field",
    "retired hurt",
    "retired out",
    "run out",
    "timed out",
}


def _is_bowler_wicket(row: pd.Series) -> bool:
    """Return whether a dismissal is credited to the bowler."""
    return bool(row["is_wicket"]) and row["dismissal_kind"] not in NON_BOWLER_DISMISSALS


# ─── Shrinkage helpers ────────────────────────────────────────────────────────

def shrunk_rate(runs: int, balls: int, prior_sr: float = 125.0, prior_weight: int = 30) -> float:
    """Bayesian-shrunk strike rate toward a prior (125 runs per 100 balls)."""
    return (runs + prior_sr / 100 * prior_weight) / (balls + prior_weight) * 100


def shrunk_average(runs: int, dismissals: int, prior_avg: float = 22.0, prior_weight: int = 5) -> float:
    """Bayesian-shrunk batting average toward a prior (22 per dismissal)."""
    return (runs + prior_avg * prior_weight) / (dismissals + prior_weight)


def shrunk_economy(runs: int, balls: int, prior_econ: float = 8.5, prior_weight: int = 24) -> float:
    """Bayesian-shrunk economy rate (runs per over) toward a prior."""
    return (runs + prior_econ / 6 * prior_weight) / (balls + prior_weight) * 6


def shrunk_wicket_rate(wickets: int, balls: int, prior: float = 1 / 25, prior_weight: int = 24) -> float:
    """Bayesian-shrunk wicket rate (wickets per ball)."""
    return (wickets + prior * prior_weight) / (balls + prior_weight)


# ─── Phase helpers ────────────────────────────────────────────────────────────

def _phase(over: int) -> str:
    if over < 6:
        return "pp"
    elif over < 15:
        return "mid"
    else:
        return "death"


# ─── Main builder ─────────────────────────────────────────────────────────────

def build_player_stats(matches: pd.DataFrame, deliveries: pd.DataFrame) -> dict[int, dict]:
    """
    Build a dict: match_id → {player_name → stats_dict}
    Stats are computed from deliveries in all matches BEFORE this match (chronological).
    Returns batting + bowling stats per player per match.
    """
    # Merge match date into deliveries
    del_with_date = deliveries.merge(
        matches[["id", "date"]],
        left_on="match_id", right_on="id",
        how="inner"
    ).sort_values(["date", "match_id"])

    # Running accumulators: keyed by player name
    bat_stats: dict[str, dict] = defaultdict(lambda: {
        "innings": 0,
        "runs": 0, "balls": 0, "dismissals": 0,
        "fours": 0, "sixes": 0,
        "pp_runs": 0, "pp_balls": 0,
        "mid_runs": 0, "mid_balls": 0,
        "death_runs": 0, "death_balls": 0,
        "recent_scores": [],   # list of (runs, balls) per innings
    })

    bowl_stats: dict[str, dict] = defaultdict(lambda: {
        "balls": 0, "runs": 0, "wickets": 0,
        "dots": 0, "fours": 0, "sixes": 0,
        "pp_balls": 0, "pp_runs": 0, "pp_wkts": 0,
        "mid_balls": 0, "mid_runs": 0, "mid_wkts": 0,
        "death_balls": 0, "death_runs": 0, "death_wkts": 0,
        "recent_spells": [],   # list of (wickets, runs, balls) per match
    })

    # Batter vs Bowler
    bvb: dict[tuple, dict] = defaultdict(lambda: {"runs": 0, "balls": 0, "dismissals": 0})

    # Player venue batting: (player, venue) → stats
    bat_venue: dict[tuple, dict] = defaultdict(lambda: {"runs": 0, "balls": 0, "dismissals": 0})
    bowl_venue: dict[tuple, dict] = defaultdict(lambda: {"balls": 0, "runs": 0, "wickets": 0})

    # Per-match-per-player snapshot
    match_player_stats: dict[int, dict] = {}

    prev_match_id = None
    current_match_deliveries = []

    ordered_match_ids = del_with_date["match_id"].unique()

    # Group by match, process in order
    for match_id, group in del_with_date.groupby("match_id", sort=False):
        venue = matches.loc[matches["id"] == match_id, "venue"].values
        venue = venue[0] if len(venue) > 0 else "Unknown"

        # Snapshot current stats BEFORE this match (for prediction use)
        snapshot = {}

        # Get all unique players in this match
        batters_in_match = group["batter"].unique()
        bowlers_in_match = group["bowler"].unique()
        all_players = set(batters_in_match) | set(bowlers_in_match)

        for player in all_players:
            bs = bat_stats[player]
            bwl = bowl_stats[player]

            # Recent batting (last 5 innings)
            recent = bs["recent_scores"][-5:]
            rec_runs = sum(r for r, b in recent)
            rec_balls = sum(b for r, b in recent)
            rec_innings = len(recent)
            rec_dismissals = min(rec_innings, bs["dismissals"])  # approximate

            snapshot[player] = {
                # Career batting
                "bat_innings": bs["innings"],
                "bat_avg": shrunk_average(bs["runs"], bs["dismissals"]),
                "bat_sr": shrunk_rate(bs["runs"], bs["balls"]),
                "bat_boundary_rate": (bs["fours"] + bs["sixes"]) / max(bs["balls"], 1),
                # Recent batting (last 5 innings)
                "bat_recent_avg": shrunk_average(rec_runs, max(rec_dismissals, 1)),
                "bat_recent_sr": shrunk_rate(rec_runs, rec_balls),
                # Phase batting
                "bat_pp_sr": shrunk_rate(bs["pp_runs"], bs["pp_balls"], prior_sr=120.0),
                "bat_mid_sr": shrunk_rate(bs["mid_runs"], bs["mid_balls"], prior_sr=120.0),
                "bat_death_sr": shrunk_rate(bs["death_runs"], bs["death_balls"], prior_sr=140.0),
                # Career bowling
                "bowl_innings": 1 if bwl["balls"] > 0 else 0,
                "bowl_econ": shrunk_economy(bwl["runs"], bwl["balls"]),
                "bowl_sr": (bwl["balls"] / max(bwl["wickets"], 1)) if bwl["wickets"] > 0 else 999,
                "bowl_wicket_rate": shrunk_wicket_rate(bwl["wickets"], bwl["balls"]),
                "bowl_dot_rate": bwl["dots"] / max(bwl["balls"], 1),
                # Phase bowling
                "bowl_pp_econ": shrunk_economy(bwl["pp_runs"], bwl["pp_balls"], prior_econ=7.5),
                "bowl_death_econ": shrunk_economy(bwl["death_runs"], bwl["death_balls"], prior_econ=10.0),
                "bowl_pp_wicket_rate": shrunk_wicket_rate(bwl["pp_wkts"], bwl["pp_balls"]),
                "bowl_death_wicket_rate": shrunk_wicket_rate(bwl["death_wkts"], bwl["death_balls"]),
                # Venue batting
                "bat_venue_avg": shrunk_average(
                    bat_venue[(player, venue)]["runs"],
                    bat_venue[(player, venue)]["dismissals"],
                ),
                "bat_venue_sr": shrunk_rate(
                    bat_venue[(player, venue)]["runs"],
                    bat_venue[(player, venue)]["balls"],
                ),
                # Venue bowling
                "bowl_venue_econ": shrunk_economy(
                    bowl_venue[(player, venue)]["runs"],
                    bowl_venue[(player, venue)]["balls"],
                ),
            }

        match_player_stats[match_id] = snapshot

        # Now UPDATE stats with this match's deliveries
        # --- Batting update ---
        innings_tracker: dict[str, dict] = defaultdict(lambda: {"runs": 0, "balls": 0, "dismissed": False})
        for _, row in group.iterrows():
            b = row["batter"]
            bowler = row["bowler"]
            runs = int(row["batsman_runs"])
            extra = int(row["extra_runs"])
            total = int(row["total_runs"])
            is_wkt = int(row["is_wicket"])
            over = int(row["over"])
            phase = _phase(over)

            bat_stats[b]["balls"] += 1
            bat_stats[b]["runs"] += runs
            if runs == 4:
                bat_stats[b]["fours"] += 1
            if runs == 6:
                bat_stats[b]["sixes"] += 1
            if phase == "pp":
                bat_stats[b]["pp_runs"] += runs
                bat_stats[b]["pp_balls"] += 1
            elif phase == "mid":
                bat_stats[b]["mid_runs"] += runs
                bat_stats[b]["mid_balls"] += 1
            else:
                bat_stats[b]["death_runs"] += runs
                bat_stats[b]["death_balls"] += 1

            innings_tracker[b]["runs"] += runs
            innings_tracker[b]["balls"] += 1

            if is_wkt and row["player_dismissed"] == b:
                bat_stats[b]["dismissals"] += 1
                innings_tracker[b]["dismissed"] = True

            # Batter vs Bowler
            bvb[(b, bowler)]["runs"] += runs
            bvb[(b, bowler)]["balls"] += 1
            if is_wkt and row["player_dismissed"] == b:
                bvb[(b, bowler)]["dismissals"] += 1

            # Venue batting
            bat_venue[(b, venue)]["runs"] += runs
            bat_venue[(b, venue)]["balls"] += 1
            if is_wkt and row["player_dismissed"] == b:
                bat_venue[(b, venue)]["dismissals"] += 1

            # --- Bowling update ---
            bwl = bowl_stats[bowler]
            if row.get("extras_type") not in ["wides", "noballs"]:
                bwl["balls"] += 1
                if runs == 0 and extra == 0:
                    bwl["dots"] += 1
            bwl["runs"] += total  # total includes extras
            bowler_wicket = _is_bowler_wicket(row)
            if bowler_wicket:
                bwl["wickets"] += 1
            if runs == 4:
                bwl["fours"] += 1
            if runs == 6:
                bwl["sixes"] += 1
            if phase == "pp":
                bwl["pp_balls"] += 1
                bwl["pp_runs"] += total
                if bowler_wicket:
                    bwl["pp_wkts"] += 1
            elif phase == "mid":
                bwl["mid_balls"] += 1
                bwl["mid_runs"] += total
                if bowler_wicket:
                    bwl["mid_wkts"] += 1
            else:
                bwl["death_balls"] += 1
                bwl["death_runs"] += total
                if bowler_wicket:
                    bwl["death_wkts"] += 1

            # Venue bowling
            bowl_venue[(bowler, venue)]["balls"] += 1
            bowl_venue[(bowler, venue)]["runs"] += total
            if bowler_wicket:
                bowl_venue[(bowler, venue)]["wickets"] += 1

        # Finalise innings for each batter (record score)
        for batter, inn in innings_tracker.items():
            bat_stats[batter]["innings"] += 1
            bat_stats[batter]["recent_scores"].append((inn["runs"], inn["balls"]))

    return match_player_stats, bvb


def build_global_player_stats(matches: pd.DataFrame, deliveries: pd.DataFrame) -> dict:
    """
    Build a FINAL (post-all-matches) stats dict: player_name → stats_dict.
    Used for display purposes only — NOT for model features (which must be leakage-free).
    Includes total career runs, innings, average, strike rate, wickets, economy, etc.
    Also includes a computed performance_score (0-100).
    """
    del_with_date = deliveries.merge(
        matches[["id", "date", "season"]],
        left_on="match_id", right_on="id",
        how="inner"
    )

    bat: dict[str, dict] = defaultdict(lambda: {
        "innings": 0, "runs": 0, "balls": 0, "dismissals": 0,
        "fours": 0, "sixes": 0, "fifties": 0, "hundreds": 0,
        "pp_runs": 0, "pp_balls": 0,
        "mid_runs": 0, "mid_balls": 0,
        "death_runs": 0, "death_balls": 0,
        "recent_scores": [],
    })
    bowl: dict[str, dict] = defaultdict(lambda: {
        "balls": 0, "runs": 0, "wickets": 0, "dots": 0,
        "pp_balls": 0, "pp_runs": 0, "pp_wkts": 0,
        "death_balls": 0, "death_runs": 0, "death_wkts": 0,
        "match_spells": [],   # (wickets, runs, balls) per match
    })
    bvb_raw: dict[tuple, dict] = defaultdict(lambda: {"runs": 0, "balls": 0, "dismissals": 0})
    teams_played: dict[str, set] = defaultdict(set)

    innings_tracker: dict = {}  # match_id → {player → {runs,balls,dismissed}}

    for match_id, grp in del_with_date.groupby("match_id", sort=False):
        inn_t: dict[str, dict] = defaultdict(lambda: {"runs": 0, "balls": 0, "dismissed": False})
        for _, row in grp.iterrows():
            b = row["batter"]
            bwlr = row["bowler"]
            runs = int(row["batsman_runs"])
            extra = int(row["extra_runs"])
            total = int(row["total_runs"])
            is_wkt = int(row["is_wicket"])
            over = int(row["over"])
            phase = _phase(over)

            bat[b]["balls"] += 1
            bat[b]["runs"] += runs
            if runs == 4: bat[b]["fours"] += 1
            if runs == 6: bat[b]["sixes"] += 1
            if phase == "pp":
                bat[b]["pp_runs"] += runs; bat[b]["pp_balls"] += 1
            elif phase == "mid":
                bat[b]["mid_runs"] += runs; bat[b]["mid_balls"] += 1
            else:
                bat[b]["death_runs"] += runs; bat[b]["death_balls"] += 1

            inn_t[b]["runs"] += runs
            inn_t[b]["balls"] += 1
            if is_wkt and row["player_dismissed"] == b:
                bat[b]["dismissals"] += 1
                inn_t[b]["dismissed"] = True

            # BvB
            bvb_raw[(b, bwlr)]["runs"] += runs
            bvb_raw[(b, bwlr)]["balls"] += 1
            if is_wkt and row["player_dismissed"] == b:
                bvb_raw[(b, bwlr)]["dismissals"] += 1

            # Bowling
            bwl = bowl[bwlr]
            if row.get("extras_type") not in ["wides", "noballs"]:
                bwl["balls"] += 1
                if runs == 0 and extra == 0: bwl["dots"] += 1
            bwl["runs"] += total
            bowler_wicket = _is_bowler_wicket(row)
            if bowler_wicket:
                bwl["wickets"] += 1
            if phase == "pp":
                bwl["pp_balls"] += 1; bwl["pp_runs"] += total
                if bowler_wicket: bwl["pp_wkts"] += 1
            else:
                bwl["death_balls"] += 1; bwl["death_runs"] += total
                if bowler_wicket: bwl["death_wkts"] += 1

        # Finalise innings
        for batter, inn in inn_t.items():
            bat[batter]["innings"] += 1
            bat[batter]["recent_scores"].append((inn["runs"], inn["balls"]))
            if inn["runs"] >= 50: bat[batter]["fifties"] += 1
            if inn["runs"] >= 100: bat[batter]["hundreds"] += 1

        # Track team associations
        batting_teams = dict(zip(grp["batter"], grp["batting_team"]))
        bowling_teams = dict(zip(grp["bowler"], grp["bowling_team"]))
        for p, t in {**batting_teams, **bowling_teams}.items():
            teams_played[p].add(t)

    # Build final stats per player
    all_players = set(bat.keys()) | set(bowl.keys())
    result = {}

    IPL_BAT_AVG = 26.0
    IPL_BAT_SR  = 130.0
    IPL_ECON    = 8.5
    IPL_WR      = 0.04

    for player in all_players:
        bs  = bat[player]
        bwl = bowl[player]

        # Recent form: last 10 innings
        recent10 = bs["recent_scores"][-10:]
        rec_runs   = sum(r for r, b in recent10)
        rec_balls  = sum(b for r, b in recent10)
        rec_inn    = len(recent10)

        # Batting computed
        b_avg = shrunk_average(bs["runs"], bs["dismissals"])
        b_sr  = shrunk_rate(bs["runs"], bs["balls"])
        b_pp_sr    = shrunk_rate(bs["pp_runs"], bs["pp_balls"], prior_sr=120.0)
        b_death_sr = shrunk_rate(bs["death_runs"], bs["death_balls"], prior_sr=140.0)
        b_recent_avg = shrunk_average(rec_runs, max(1, bs["dismissals"]) if rec_inn < 3 else max(1, rec_inn - sum(1 for r,_ in recent10 if r > 0) // 2))
        b_recent_sr  = shrunk_rate(rec_runs, rec_balls)

        # Bowling computed
        bwl_econ  = shrunk_economy(bwl["runs"], bwl["balls"])
        bwl_wr    = shrunk_wicket_rate(bwl["wickets"], bwl["balls"])
        bwl_pp_e  = shrunk_economy(bwl["pp_runs"], bwl["pp_balls"], prior_econ=7.5)
        bwl_d_e   = shrunk_economy(bwl["death_runs"], bwl["death_balls"], prior_econ=10.0)
        bwl_dot   = bwl["dots"] / max(bwl["balls"], 1)

        is_genuine_allrounder = (bs["runs"] >= 250 and bwl["wickets"] >= 10 and bwl["balls"] >= 180)
        is_primary_bowler = (bwl["balls"] >= 120 and (bs["runs"] < 350 or bwl["wickets"] >= 25))
        is_primary_batter = (bs["innings"] >= 5 and bwl["balls"] < 250)

        # Performance score (0-100): benchmarked against IPL career standards
        bat_score = 45.0
        bowl_score = 45.0
        if bs["innings"] >= 1:
            avg_z = min(max((b_avg - IPL_BAT_AVG) / 10.0, -2.5), 3.0)
            sr_z = min(max((b_sr - IPL_BAT_SR) / 15.0, -2.5), 3.0)
            vol_boost = min(bs["runs"] / 1000.0 * 2.0, 10.0)
            bat_score = min(100.0, max(20.0, 50.0 + avg_z * 15.0 + sr_z * 10.0 + vol_boost))

        if bwl["balls"] >= 18:
            econ_z = min(max((IPL_ECON - bwl_econ) / 1.0, -2.5), 3.0)
            wr_z = min(max((bwl_wr - IPL_WR) / 0.01, -2.5), 3.0)
            vol_boost = min(bwl["wickets"] / 20.0 * 2.0, 10.0)
            bowl_score = min(100.0, max(20.0, 50.0 + econ_z * 15.0 + wr_z * 10.0 + vol_boost))

        if is_genuine_allrounder:
            perf_score = round(max(bat_score, bowl_score) * 0.6 + min(bat_score, bowl_score) * 0.4 + 4)
        elif is_primary_bowler:
            perf_score = round(bowl_score)
        elif is_primary_batter:
            perf_score = round(bat_score)
        elif bs["innings"] >= 1 and bwl["balls"] < 30:
            perf_score = round(bat_score)
        elif bwl["balls"] >= 18 and bs["runs"] < 50:
            perf_score = round(bowl_score)
        else:
            perf_score = round(max(bat_score, bowl_score))

        perf_score = min(99, max(25, perf_score))

        is_batter = is_primary_batter or is_genuine_allrounder or (bs["innings"] >= 3)
        is_bowler = is_primary_bowler or is_genuine_allrounder or (bwl["balls"] >= 30)

        result[player] = {
            # Batting
            "bat_innings":      bs["innings"],
            "bat_runs":         bs["runs"],
            "bat_avg":          round(b_avg, 2),
            "bat_sr":           round(b_sr, 1),
            "bat_fours":        bs["fours"],
            "bat_sixes":        bs["sixes"],
            "bat_fifties":      bs["fifties"],
            "bat_hundreds":     bs["hundreds"],
            "bat_pp_sr":        round(b_pp_sr, 1),
            "bat_death_sr":     round(b_death_sr, 1),
            "bat_recent_avg":   round(b_recent_avg, 2),
            "bat_recent_sr":    round(b_recent_sr, 1),
            "bat_boundary_rate": round(bs["fours"] + bs["sixes"]) / max(bs["balls"], 1),
            "bat_hs":           max((r for r, _ in bs["recent_scores"]), default=0),
            # Bowling
            "bowl_balls":       bwl["balls"],
            "bowl_wickets":     bwl["wickets"],
            "bowl_econ":        round(bwl_econ, 2),
            "bowl_sr":          round(bwl["balls"] / max(bwl["wickets"], 1), 1),
            "bowl_wicket_rate": bwl_wr,
            "bowl_dot_rate":    round(bwl_dot, 3),
            "bowl_pp_econ":     round(bwl_pp_e, 2),
            "bowl_death_econ":  round(bwl_d_e, 2),
            "bowl_innings":     1 if bwl["balls"] > 0 else 0,
            # Role flags
            "is_batter":   is_batter,
            "is_bowler":   is_bowler,
            "is_allrounder": is_genuine_allrounder,
            # Performance score
            "perf_score":  perf_score,
        }

    return result


def build_venue_player_stats(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    venue: str,
) -> dict[str, dict]:
    """Build batting and credited bowling stats for one venue."""
    venue_match_ids = set(matches.loc[matches["venue"] == venue, "id"])
    venue_deliveries = deliveries[deliveries["match_id"].isin(venue_match_ids)]
    stats: dict[str, dict] = defaultdict(lambda: {
        "bat_runs": 0,
        "bat_balls": 0,
        "bat_dismissals": 0,
        "bat_innings": set(),
        "bowl_runs": 0,
        "bowl_balls": 0,
        "bowl_wickets": 0,
    })

    for _, row in venue_deliveries.iterrows():
        batter = row["batter"]
        bowler = row["bowler"]
        match_id = row["match_id"]
        runs = int(row["batsman_runs"])
        total = int(row["total_runs"])
        is_bowler_wicket = _is_bowler_wicket(row)

        batter_stats = stats[batter]
        batter_stats["bat_runs"] += runs
        batter_stats["bat_balls"] += 1
        batter_stats["bat_innings"].add(match_id)
        if row["is_wicket"] and row["player_dismissed"] == batter:
            batter_stats["bat_dismissals"] += 1

        bowler_stats = stats[bowler]
        bowler_stats["bowl_runs"] += total
        if row["extras_type"] not in ["wides", "noballs"]:
            bowler_stats["bowl_balls"] += 1
        if is_bowler_wicket:
            bowler_stats["bowl_wickets"] += 1

    result = {}
    for player, values in stats.items():
        result[player] = {
            "bat_venue_avg": values["bat_runs"] / max(values["bat_dismissals"], 1),
            "bat_venue_sr": values["bat_runs"] / max(values["bat_balls"], 1) * 100,
            "bat_venue_innings": len(values["bat_innings"]),
            "bowl_venue_econ": values["bowl_runs"] / max(values["bowl_balls"], 1) * 6,
            "bowl_venue_wickets": values["bowl_wickets"],
            "bowl_venue_balls": values["bowl_balls"],
        }
    return result


def get_bvb_stats(bvb: dict, batter: str, bowler: str) -> dict:
    """
    Return professional Bayesian-shrunk and net-impact batter-vs-bowler matchup stats.
    Incorporates both scoring rate and dismissal risk for balanced analysis.
    """
    key = (batter, bowler)
    d = bvb.get(key, {"runs": 0, "balls": 0, "dismissals": 0})
    balls = d["balls"]
    runs = d["runs"]
    dismissals = d["dismissals"]

    # Shrunk metrics with priors
    sr = shrunk_rate(runs, balls, prior_sr=125.0, prior_weight=20)
    avg = shrunk_average(runs, dismissals, prior_avg=22.0, prior_weight=3)

    # Net BvB Impact Index (Runs - 18 * Dismissals) / Balls * 100
    # Average T20 IPL benchmark is ~+40.0 index points (1.30 rpo - 0.05 wkts * 18)
    if balls > 0:
        net_impact = ((runs - 18.0 * dismissals) / balls) * 100
        raw_sr = (runs / balls) * 100
        raw_avg = (runs / dismissals) if dismissals > 0 else float(runs)
    else:
        net_impact = 40.0
        raw_sr = 125.0
        raw_avg = 22.0

    return {
        "bvb_balls":       balls,
        "bvb_runs":        runs,
        "bvb_dismissals":  dismissals,
        "bvb_sr":          sr,
        "bvb_avg":         avg,
        "bvb_raw_sr":      raw_sr,
        "bvb_raw_avg":     raw_avg,
        "bvb_net_impact":  net_impact,
    }

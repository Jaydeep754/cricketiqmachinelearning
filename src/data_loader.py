"""
src/data_loader.py
Loads and normalises the CricketIQ datasets.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Pune Warriors": "Rising Pune Supergiant",
}

VENUE_NAME_MAP = {
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "Eden Gardens, Kolkata": "Eden Gardens",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium",
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium",
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium, Ahmedabad",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
}


def load_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalise matches + deliveries CSVs."""
    matches = pd.read_csv(base_dir / "data" / "matches.csv")
    deliveries = pd.read_csv(base_dir / "data" / "deliveries.csv")

    # Normalise team names
    team_cols = ["team1", "team2", "toss_winner", "winner"]
    for col in team_cols:
        matches[col] = matches[col].replace(TEAM_NAME_MAP)
    for col in ["batting_team", "bowling_team"]:
        deliveries[col] = deliveries[col].replace(TEAM_NAME_MAP)

    # Normalise venue names
    matches["venue"] = matches["venue"].replace(VENUE_NAME_MAP)

    # Parse dates
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values(["date", "id"]).reset_index(drop=True)

    # Drop no-result matches (winner is NaN) — 5 matches
    matches = matches.dropna(subset=["winner"]).reset_index(drop=True)

    return matches, deliveries


CURRENT_TEAMS = sorted([
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
])

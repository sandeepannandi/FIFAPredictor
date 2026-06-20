"""
Utility functions for the FIFA World Cup 2026 Predictor.
Contains constants, ELO computation, and helper functions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_FIXTURES = PROJECT_ROOT / "data" / "fixtures"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
for d in [DATA_RAW, DATA_PROCESSED, DATA_FIXTURES, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- ELO Constants ----------
ELO_K = 32
ELO_HOME_ADV = 100
ELO_DRAW_WEIGHT = 0.5
ELO_MARGIN_MULTIPLIER = 1.0
ELO_INITIAL = 1500


def compute_elo_ratings(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute simple ELO ratings for all teams over time from match results.

    Returns (df_with_elo, final_ratings_df).
    """
    df = results_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    elo_dict: dict[str, float] = {}
    elo_home_list = []
    elo_away_list = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        neutral = row.get("neutral", False)

        if home not in elo_dict:
            elo_dict[home] = ELO_INITIAL
        if away not in elo_dict:
            elo_dict[away] = ELO_INITIAL

        elo_home = elo_dict[home]
        elo_away = elo_dict[away]

        home_adv = 0 if neutral else ELO_HOME_ADV
        expected_home = 1.0 / (1.0 + 10.0 ** ((elo_away - (elo_home + home_adv)) / 400.0))
        expected_away = 1.0 - expected_home

        goal_diff = abs(home_score - away_score)
        if goal_diff <= 1:
            margin_mul = 1.0
        elif goal_diff == 2:
            margin_mul = 1.5
        else:
            margin_mul = (11.0 + goal_diff) / 8.0

        if home_score > away_score:
            actual_home, actual_away = 1.0, 0.0
        elif home_score < away_score:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        k = ELO_K * margin_mul
        elo_dict[home] += k * (actual_home - expected_home)
        elo_dict[away] += k * (actual_away - expected_away)

        elo_home_list.append(elo_dict[home])
        elo_away_list.append(elo_dict[away])

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list

    final_ratings = pd.DataFrame(
        list(elo_dict.items()), columns=["team", "elo_rating"]
    ).sort_values("elo_rating", ascending=False).reset_index(drop=True)

    return df, final_ratings


def get_tournament_weight(tournament: str) -> float:
    """
    Assign importance weight to a tournament type.
    """
    t = tournament.lower() if tournament else ""

    if "friendly" in t:
        return 1.0

    # World Cup qualifiers (check before generic 'world cup' and 'qualifier')
    if "world cup qualification" in t:
        return 3.0

    # World Cup proper (highest importance)
    if "world cup" in t:
        return 4.0

    # Other qualifiers
    if any(x in t for x in ["qualifier", "qualification"]):
        return 2.0

    # Continental tournament knockouts
    if any(x in t for x in ["semi", "final", "quarter", "round"]):
        return 3.5

    # Continental tournaments (group stage)
    continental = [
        "africa cup of nations", "asian cup", "european championship",
        "copa america", "gold cup", "nations league", "concacaf",
        "ofa nations cup"
    ]
    if any(x in t for x in continental):
        return 3.0

    return 1.5


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)


# Team name normalization map to handle inconsistencies across datasets
TEAM_NAME_NORMALIZATION = {
    "United States": "USA",
    "USA": "USA",
    "US": "USA",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Iran": "Iran",
    "IR Iran": "Iran",
    "Ivory Coast": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia": "Bosnia and Herzegovina",
    "Czech Republic": "Czech Republic",
    "Czechia": "Czech Republic",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Congo-Kinshasa": "DR Congo",
    "Cape Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Curacao": "Curacao",
    "Curaçao": "Curacao",
    "CuraÃ§ao": "Curacao",
    "New Zealand": "New Zealand",
    "Scotland": "Scotland",
    "Turkey": "Turkey",
    "Turkiye": "Turkey",
    "China": "China PR",
    "China PR": "China PR",
    "Russia": "Russia",
}


def normalize_team_name(name: str) -> str:
    """Normalize team names across datasets."""
    return TEAM_NAME_NORMALIZATION.get(str(name).strip(), str(name).strip())

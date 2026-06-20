"""
feature_engineering.py -- Build per-match feature vectors from cleaned results.

Features computed for each match:
  - ELO rating difference
  - Recent form (points from last 10 matches)
  - Goals scored / conceded averages (last 10 matches)
  - Head-to-head win rate (last 10 meetings)
  - Home / away / neutral venue flag
  - Tournament importance weight
  - Days since last match (fatigue proxy)

All features are computed strictly from data before the match date
(no look-ahead bias).
"""

import pandas as pd
import numpy as np
from src.utils import compute_elo_ratings, get_tournament_weight, DATA_PROCESSED


def _build_team_history(matches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build a dictionary mapping team name to a DataFrame of their past matches
    sorted by date, each row from that team's perspective.
    """
    home = matches[["date", "home_team", "away_team", "home_score", "away_score",
                     "neutral", "tournament"]].copy()
    home.columns = ["date", "team", "opponent", "goals_for", "goals_against",
                     "neutral", "tournament"]
    home["venue"] = "home"

    away = matches[["date", "away_team", "home_team", "away_score", "home_score",
                     "neutral", "tournament"]].copy()
    away.columns = ["date", "team", "opponent", "goals_for", "goals_against",
                     "neutral", "tournament"]
    away["venue"] = "away"

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches = team_matches.sort_values(["team", "date"]).reset_index(drop=True)

    history: dict[str, pd.DataFrame] = {}
    for team, grp in team_matches.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        history[team] = grp
    return history


def _recent_form(team_df: pd.DataFrame, idx: int, n: int = 10) -> float:
    """Average points (3/1/0) from the last n matches before idx."""
    if idx < 1:
        return 0.0
    start = max(0, idx - n)
    recent = team_df.iloc[start:idx]
    if len(recent) == 0:
        return 0.0
    pts = 0
    for _, r in recent.iterrows():
        if r["goals_for"] > r["goals_against"]:
            pts += 3
        elif r["goals_for"] == r["goals_against"]:
            pts += 1
    return pts / len(recent)


def _recent_goals_scored(team_df: pd.DataFrame, idx: int, n: int = 10) -> float:
    """Average goals scored in last n matches."""
    if idx < 1:
        return 0.0
    start = max(0, idx - n)
    recent = team_df.iloc[start:idx]
    if len(recent) == 0:
        return 0.0
    return recent["goals_for"].mean()


def _recent_goals_conceded(team_df: pd.DataFrame, idx: int, n: int = 10) -> float:
    """Average goals conceded in last n matches."""
    if idx < 1:
        return 0.0
    start = max(0, idx - n)
    recent = team_df.iloc[start:idx]
    if len(recent) == 0:
        return 0.0
    return recent["goals_against"].mean()


def _h2h_win_rate(team_df: pd.DataFrame, opponent: str, idx: int, n: int = 10) -> float:
    """Head-to-head win rate against a specific opponent in last n meetings."""
    if idx < 1:
        return 0.0
    recent = team_df.iloc[:idx]
    meetings = recent[recent["opponent"] == opponent].tail(n)
    if len(meetings) == 0:
        return 0.0
    wins = (meetings["goals_for"] > meetings["goals_against"]).sum()
    return wins / len(meetings)


def _days_since_last_match(team_df: pd.DataFrame, idx: int) -> float:
    """Days since the team's previous match. Returns 0 if no prior match."""
    if idx < 1:
        return 0.0
    prev_date = team_df.iloc[idx - 1]["date"]
    curr_date = team_df.iloc[idx]["date"]
    return (curr_date - prev_date).days


def build_features(matches: pd.DataFrame, cache: bool = True) -> pd.DataFrame:
    """
    Build feature vectors for each match.
    Returns a DataFrame with features ready for model training.
    """
    if cache and DATA_PROCESSED.joinpath("matches_features.parquet").exists():
        print("[OK] Loading cached feature vectors...")
        return pd.read_parquet(DATA_PROCESSED / "matches_features.parquet")

    print("[..] Computing ELO ratings...")
    matches_with_elo, final_elo = compute_elo_ratings(matches)
    final_elo.to_parquet(DATA_PROCESSED / "final_elo.parquet", index=False)

    print(f"[..] Building features for {len(matches_with_elo):,} matches...")
    history = _build_team_history(matches_with_elo)

    features_list = []

    for i, row in matches_with_elo.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in history or away not in history:
            continue

        home_df = history[home].reset_index(drop=True)
        away_df = history[away].reset_index(drop=True)

        # Find the index of this match in each team's history
        home_idx = home_df[
            (home_df["date"] == row["date"]) &
            (home_df["opponent"] == away)
        ].index
        away_idx = away_df[
            (away_df["date"] == row["date"]) &
            (away_df["opponent"] == home)
        ].index

        if len(home_idx) == 0 or len(away_idx) == 0:
            continue
        home_idx = home_idx[0]
        away_idx = away_idx[0]

        # -- Target (from home perspective) --
        if row["home_score"] > row["away_score"]:
            target = "Win"
        elif row["home_score"] == row["away_score"]:
            target = "Draw"
        else:
            target = "Loss"

        # -- Features --
        elo_diff = row["elo_home"] - row["elo_away"]

        home_form = _recent_form(home_df, home_idx)
        away_form = _recent_form(away_df, away_idx)

        home_gs = _recent_goals_scored(home_df, home_idx)
        home_gc = _recent_goals_conceded(home_df, home_idx)
        away_gs = _recent_goals_scored(away_df, away_idx)
        away_gc = _recent_goals_conceded(away_df, away_idx)

        h2h_home = _h2h_win_rate(home_df, away, home_idx)
        h2h_away = _h2h_win_rate(away_df, home, away_idx)

        venue_home = 1.0 if not row.get("neutral", False) else 0.0
        venue_neutral = 1.0 if row.get("neutral", False) else 0.0

        tourney_weight = get_tournament_weight(row["tournament"])

        days_since_home = _days_since_last_match(home_df, home_idx)
        days_since_away = _days_since_last_match(away_df, away_idx)

        features_list.append({
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "tournament_weight": tourney_weight,
            "elo_diff": elo_diff,
            "home_form": home_form,
            "away_form": away_form,
            "home_goals_scored_avg": home_gs,
            "home_goals_conceded_avg": home_gc,
            "away_goals_scored_avg": away_gs,
            "away_goals_conceded_avg": away_gc,
            "h2h_home_win_rate": h2h_home,
            "h2h_away_win_rate": h2h_away,
            "venue_home": venue_home,
            "venue_neutral": venue_neutral,
            "days_since_home": days_since_home,
            "days_since_away": days_since_away,
            "target": target,
        })

    result = pd.DataFrame(features_list)
    result = result.sort_values("date").reset_index(drop=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    result.to_parquet(DATA_PROCESSED / "matches_features.parquet", index=False)

    print(f"[OK] Feature matrix: {len(result):,} rows x {len(result.columns)} columns")
    print(f"     Target distribution:\n{result['target'].value_counts(normalize=True).to_string()}")

    return result

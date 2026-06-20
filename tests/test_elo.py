# pyrefly: ignore [missing-import]
import pytest
import pandas as pd
from src.utils import compute_elo_ratings, ELO_INITIAL

def test_elo_computation_basic():
    """Test that ELO ratings update correctly after a single match."""
    results = pd.DataFrame([{
        "date": "2024-01-01",
        "home_team": "TeamA",
        "away_team": "TeamB",
        "home_score": 2,
        "away_score": 0,
        "neutral": True
    }])
    
    df_elo, final_ratings = compute_elo_ratings(results)
    
    # TeamA won, rating should be > INITIAL
    # TeamB lost, rating should be < INITIAL
    rating_a = final_ratings[final_ratings["team"] == "TeamA"]["elo_rating"].iloc[0]
    rating_b = final_ratings[final_ratings["team"] == "TeamB"]["elo_rating"].iloc[0]
    
    assert rating_a > ELO_INITIAL
    assert rating_b < ELO_INITIAL
    assert rating_a + rating_b == pytest.approx(2 * ELO_INITIAL)

def test_elo_draw():
    """Test that ELO ratings stay balanced after a draw between equal teams."""
    results = pd.DataFrame([{
        "date": "2024-01-01",
        "home_team": "TeamA",
        "away_team": "TeamB",
        "home_score": 1,
        "away_score": 1,
        "neutral": True
    }])
    
    df_elo, final_ratings = compute_elo_ratings(results)
    
    rating_a = final_ratings[final_ratings["team"] == "TeamA"]["elo_rating"].iloc[0]
    rating_b = final_ratings[final_ratings["team"] == "TeamB"]["elo_rating"].iloc[0]
    
    assert rating_a == pytest.approx(ELO_INITIAL)
    assert rating_b == pytest.approx(ELO_INITIAL)

def test_elo_home_advantage():
    """Test that home advantage affects expected outcome."""
    # Team A at home (boosted) against Team B
    # Even if they draw, Team A "underperformed" compared to expectations
    results = pd.DataFrame([{
        "date": "2024-01-01",
        "home_team": "TeamA",
        "away_team": "TeamB",
        "home_score": 1,
        "away_score": 1,
        "neutral": False
    }])
    
    df_elo, final_ratings = compute_elo_ratings(results)
    
    rating_a = final_ratings[final_ratings["team"] == "TeamA"]["elo_rating"].iloc[0]
    rating_b = final_ratings[final_ratings["team"] == "TeamB"]["elo_rating"].iloc[0]
    
    # TeamA should lose some points because they were expected to win due to home adv
    assert rating_a < ELO_INITIAL
    assert rating_b > ELO_INITIAL

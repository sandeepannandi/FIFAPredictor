# pyrefly: ignore [missing-import]
import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.simulate import TournamentSimulator

@pytest.fixture
def mock_config():
    return {
        "groups": {
            "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
            "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"]
        }
    }

@pytest.fixture
def mock_model():
    model = MagicMock()
    # Mock predict_proba to return [Win, Draw, Loss] probabilities
    model.predict_proba.return_value = [[0.4, 0.3, 0.3]]
    return model

def test_simulator_init(mock_model, mock_config):
    sim = TournamentSimulator(mock_model, mock_config)
    assert len(sim.all_teams) == 8
    assert "Mexico" in sim.stats

def test_select_knockout_teams(mock_model, mock_config):
    sim = TournamentSimulator(mock_model, mock_config)
    
    # Mock group results
    # (team, points, GD, GF, GA, wins)
    group_results = {
        "A": [
            ("Mexico", 9, 5, 6, 1, 3),        # 1st
            ("South Korea", 6, 2, 4, 2, 2),  # 2nd
            ("Czech Republic", 3, -1, 2, 3, 1), # 3rd
            ("South Africa", 0, -6, 1, 7, 0) # 4th
        ],
        "B": [
            ("Switzerland", 7, 4, 5, 1, 2),   # 1st
            ("Canada", 5, 1, 3, 2, 1),        # 2nd
            ("Qatar", 4, 0, 2, 2, 1),         # 3rd
            ("Bosnia and Herzegovina", 0, -5, 1, 6, 0) # 4th
        ]
    }
    
    # In our mock, we only have 2 groups, so BEST_THIRD_PLACES (8)
    # will actually take all third places if available.
    advancing = sim._select_knockout_teams(group_results)
    
    # Top 2 from A: Mexico, South Korea
    # Top 2 from B: Switzerland, Canada
    # 3rd places: Czech Republic, Qatar
    assert "Mexico" in advancing
    assert "South Korea" in advancing
    assert "Switzerland" in advancing
    assert "Canada" in advancing
    assert "Czech Republic" in advancing
    assert "Qatar" in advancing

def test_knockout_match_no_draws(mock_model, mock_config):
    sim = TournamentSimulator(mock_model, mock_config)
    winner = sim._simulate_knockout_match("TeamA", "TeamB")
    assert winner in ["TeamA", "TeamB"]

def test_group_points_calculation(mock_model, mock_config):
    sim = TournamentSimulator(mock_model, mock_config)
    
    # Mock _sample_group_match to always return 1-0 win for home
    sim._sample_group_match = MagicMock(return_value=(1, 0))
    
    results = sim._simulate_group_stage()
    # Mexico (index 0) played 3 matches in Group A, should have 9 points
    mexico_stats = results["A"][0]
    assert mexico_stats[0] == "Mexico"
    assert mexico_stats[1] == 9 # points

"""
simulate.py -- Monte Carlo tournament simulation for the 2026 World Cup.

Flow:
  1. Load group stage fixtures (12 groups x 4 teams = 48 teams)
  2. For each group match, get Win/Draw/Loss probabilities from the model
  3. Sample outcomes, compute group standings, advance top-2 + 8 best 3rd-placed
  4. Knockout rounds: Round of 32 -> Round of 16 -> QF -> SF -> Final
  5. Repeat N times and aggregate statistics
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.train_model import load_model, FEATURE_COLS, TARGET_MAP
from src.data_loader import load_group_config
from src.utils import set_random_seed, DATA_FIXTURES, OUTPUTS_DIR

# -- Constants ------------------------------------------------------------
N_SIMULATIONS = 10_000
GROUP_POINTS_WIN = 3
GROUP_POINTS_DRAW = 1
BEST_THIRD_PLACES = 8

TIEBREAKER_COLS = ["points", "goal_diff", "goals_for", "fifa_ranking_proxy"]


class TournamentSimulator:
    """
    Monte Carlo simulator for the 2026 FIFA World Cup.
    """

    def __init__(self, model, group_config: dict, elo_ratings: dict[str, float] = None):
        self.model = model
        self.groups = group_config["groups"]
        self.all_teams = []
        for grp_teams in self.groups.values():
            self.all_teams.extend(grp_teams)

        # Track results across all simulations
        self.stats = {team: {"r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "win": 0}
                      for team in self.all_teams}

        # Real ELO ratings (fallback to 1500 if not provided or team missing)
        self.elo_ratings = elo_ratings or {}
        self.default_elo = 1500.0

    def _get_team_rating(self, team: str) -> float:
        """Get the ELO rating for a team, or fallback to default."""
        return self.elo_ratings.get(team, self.default_elo)

    def _predict_match(self, home: str, away: str, neutral: bool = True) -> dict[str, float]:
        """
        Predict probabilities for a hypothetical match using the trained model.
        Features are estimated from the team ELO ratings since we cannot
        compute rolling features for future tournament matches.
        """
        r_home = self._get_team_rating(home)
        r_away = self._get_team_rating(away)

        # Map ELO ratings into approximately the same feature space as training:
        # elo_diff is naturally the same.
        # form, goals_scored, etc. are proxied by the ELO rating itself.
        elo_diff = r_home - r_away
        
        # Heuristic scaling for features based on ELO
        # (normalized around 1500-1800 range typically)
        form_proxy_home = (r_home - 1000) / 1000.0  # e.g., 1800 -> 0.8
        form_proxy_away = (r_away - 1000) / 1000.0
        
        features = np.array([[
            elo_diff,                        # elo_diff
            max(0, form_proxy_home),         # home_form
            max(0, form_proxy_away),         # away_form
            1.5 + (r_home - 1500) / 400.0,   # home_goals_scored_avg
            1.2 - (r_home - 1500) / 800.0,   # home_goals_conceded_avg
            1.5 + (r_away - 1500) / 400.0,   # away_goals_scored_avg
            1.2 - (r_away - 1500) / 800.0,   # away_goals_conceded_avg
            0.5,                             # h2h_home_win_rate (neutral default)
            0.4,                             # h2h_away_win_rate (neutral default)
            0.0,                             # venue_home
            1.0,                             # venue_neutral
            5.0,                             # days_since_home
            5.0,                             # days_since_away
            4.0,                             # tournament_weight (World Cup)
        ]], dtype=np.float64)

        proba = self.model.predict_proba(features)[0]
        return {
            "home_win": float(proba[0]),
            "draw": float(proba[1]),
            "away_win": float(proba[2]),
        }

    def _sample_group_match(self, home: str, away: str) -> tuple[int, int]:
        """Sample a match result using model probabilities and Poisson goal distribution."""
        probs = self._predict_match(home, away, neutral=True)
        outcome = np.random.choice(
            ["home_win", "draw", "away_win"],
            p=[probs["home_win"], probs["draw"], probs["away_win"]]
        )

        if outcome == "home_win":
            home_g = np.random.poisson(1.8)
            away_g = np.random.poisson(0.6)
            if home_g <= away_g:
                home_g = away_g + max(1, np.random.poisson(1))
        elif outcome == "draw":
            goals = np.random.poisson(1.2)
            home_g = away_g = goals
        else:
            home_g = np.random.poisson(0.6)
            away_g = np.random.poisson(1.8)
            if away_g <= home_g:
                away_g = home_g + max(1, np.random.poisson(1))

        return int(home_g), int(away_g)

    def _simulate_group_stage(self) -> dict[str, list[tuple]]:
        """
        Simulate all group matches.
        Returns: {group_label: [(team, points, GD, GF, GA, wins), ...]}
        Sorted by group standing (position 0 = group winner).
        """
        group_results = {}

        for group_label, teams in self.groups.items():
            standings: dict[str, dict] = {t: {"points": 0, "gf": 0, "ga": 0, "gd": 0, "wins": 0}
                                           for t in teams}

            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    home, away = teams[i], teams[j]
                    home_g, away_g = self._sample_group_match(home, away)

                    standings[home]["gf"] += home_g
                    standings[home]["ga"] += away_g
                    standings[away]["gf"] += away_g
                    standings[away]["ga"] += home_g

                    if home_g > away_g:
                        standings[home]["points"] += GROUP_POINTS_WIN
                        standings[home]["wins"] += 1
                    elif home_g == away_g:
                        standings[home]["points"] += GROUP_POINTS_DRAW
                        standings[away]["points"] += GROUP_POINTS_DRAW
                    else:
                        standings[away]["points"] += GROUP_POINTS_WIN
                        standings[away]["wins"] += 1

            # Compute goal difference for each team
            for t in teams:
                standings[t]["gd"] = standings[t]["gf"] - standings[t]["ga"]

            # Sort by points -> GD -> GF -> ranking proxy
            sorted_teams = sorted(
                teams,
                key=lambda t: (
                    standings[t]["points"],
                    standings[t]["gd"],
                    standings[t]["gf"],
                    self._get_team_rating(t),
                ),
                reverse=True,
            )

            group_results[group_label] = [
                (t, standings[t]["points"], standings[t]["gd"],
                 standings[t]["gf"], standings[t]["ga"], standings[t]["wins"])
                for t in sorted_teams
            ]

        return group_results

    def _select_knockout_teams(self, group_results: dict) -> list[str]:
        """
        From group results, select 32 teams advancing to R32.
        Top 2 from each group (24 teams) + 8 best third-placed teams.
        """
        first_place = []
        second_place = []

        for group_label, standings in group_results.items():
            first_place.append((group_label, standings[0][0]))
            second_place.append((group_label, standings[1][0]))

        # Rank third-placed teams
        third_detailed = []
        for group_label, standings in group_results.items():
            t3 = standings[2]
            third_detailed.append({
                "group": group_label,
                "team": t3[0],
                "points": t3[1],
                "gd": t3[2],
                "gf": t3[3],
                "ranking": self._get_team_rating(t3[0]),
            })

        third_df = pd.DataFrame(third_detailed)
        third_df = third_df.sort_values(
            ["points", "gd", "gf", "ranking"],
            ascending=False
        ).head(BEST_THIRD_PLACES)

        advancing_third = third_df["team"].tolist()

        advancing = ([t[1] for t in first_place] +
                     [t[1] for t in second_place] +
                     advancing_third)
        return advancing

    def _simulate_knockout_match(self, team_a: str, team_b: str) -> str:
        """
        Simulate a single knockout match. No draws allowed:
        normalize win probabilities and sample the winner.
        """
        probs = self._predict_match(team_a, team_b, neutral=True)
        p_win_a = probs["home_win"]
        p_draw = probs["draw"]
        p_win_b = probs["away_win"]

        # Redistribute draw probability proportionally
        p_win_a_adj = p_win_a + p_draw / 2
        p_win_b_adj = p_win_b + p_draw / 2
        total = p_win_a_adj + p_win_b_adj
        p_win_a_adj /= total
        p_win_b_adj /= total

        winner = np.random.choice([team_a, team_b], p=[p_win_a_adj, p_win_b_adj])
        return winner

    def _build_knockout_bracket(self, advancing_teams: list[str]) -> list[list[str]]:
        """
        Build knockout bracket pairings for R32.

        32 teams into 16 matches:
        - 8 group winners (strongest) vs 8 best third-placed teams
        - 4 group winners (weakest) vs 4 lowest-ranked runners-up
        - Remaining 8 runners-up paired against each other
        """
        winners = advancing_teams[:12]
        runners_up = advancing_teams[12:24]
        third_placed = advancing_teams[24:]

        runners_up_sorted = sorted(
            runners_up,
            key=lambda t: self._get_team_rating(t)
        )

        matches = []

        # Strongest 8 winners vs 8 third-placed teams
        winners_sorted = sorted(
            winners,
            key=lambda t: self._get_team_rating(t),
            reverse=True
        )
        for i in range(8):
            matches.append([winners_sorted[i], third_placed[i]])

        # Weakest 4 winners vs weakest 4 runners-up
        remaining_winners = winners_sorted[8:]
        weakest_runners_up = runners_up_sorted[:4]
        for i in range(4):
            matches.append([remaining_winners[i], weakest_runners_up[i]])

        # Remaining 8 runners-up paired against each other
        remaining_runners_up = runners_up_sorted[4:]
        for i in range(4):
            matches.append([remaining_runners_up[i], remaining_runners_up[i + 4]])

        return matches

    def _simulate_knockout_round(self, teams: list[str]) -> list[str]:
        """Generic: pair teams sequentially and return winners."""
        winners = []
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                winners.append(self._simulate_knockout_match(teams[i], teams[i + 1]))
        return winners

    def run_single_simulation(self) -> str:
        """Run one full tournament simulation. Returns the champion."""
        group_results = self._simulate_group_stage()
        advancing = self._select_knockout_teams(group_results)
        r32_matches = self._build_knockout_bracket(advancing)

        r16_teams = self._simulate_r32(r32_matches)

        qf_teams = self._simulate_knockout_round(r16_teams)
        sf_teams = self._simulate_knockout_round(qf_teams)
        finalists = self._simulate_knockout_round(sf_teams)
        champion = self._simulate_knockout_match(finalists[0], finalists[1])

        advancing_set = set(advancing)
        r16_set = set(r16_teams)
        qf_set = set(qf_teams)
        sf_set = set(sf_teams)
        final_set = set(finalists)

        for team in advancing_set:
            self.stats[team]["r32"] += 1
        for team in r16_set:
            self.stats[team]["r16"] += 1
        for team in qf_set:
            self.stats[team]["qf"] += 1
        for team in sf_set:
            self.stats[team]["sf"] += 1
        for team in final_set:
            self.stats[team]["final"] += 1
        self.stats[champion]["win"] += 1

        return champion

    def _simulate_r32(self, matches: list[list[str]]) -> list[str]:
        """Simulate Round of 32 matches, return winners (who advance to R16)."""
        winners = []
        for match in matches:
            winners.append(self._simulate_knockout_match(match[0], match[1]))
        return winners

    def run(self, n_simulations: int = N_SIMULATIONS) -> pd.DataFrame:
        """Run N simulations and aggregate results."""
        set_random_seed(42)

        print(f"[..] Running {n_simulations:,} tournament simulations...")
        for i in range(n_simulations):
            if (i + 1) % 1000 == 0:
                print(f"    {i + 1:,} / {n_simulations:,}...")
            self.run_single_simulation()

        print(f"[OK] {n_simulations:,} simulations complete.")

        results = []
        for team in self.all_teams:
            s = self.stats[team]
            results.append({
                "team": team,
                "pct_r32": round(s["r32"] / n_simulations * 100, 2),
                "pct_r16": round(s["r16"] / n_simulations * 100, 2),
                "pct_qf": round(s["qf"] / n_simulations * 100, 2),
                "pct_sf": round(s["sf"] / n_simulations * 100, 2),
                "pct_final": round(s["final"] / n_simulations * 100, 2),
                "pct_win": round(s["win"] / n_simulations * 100, 2),
            })

        results_df = pd.DataFrame(results).sort_values("pct_win", ascending=False)
        results_df.to_csv(OUTPUTS_DIR / "win_probabilities.csv", index=False)
        print(f"[OK] Results saved to {OUTPUTS_DIR / 'win_probabilities.csv'}")
        return results_df

    @staticmethod
    def plot_results(results_df: pd.DataFrame, top_n: int = 10):
        """Plot bar chart of top N teams by win probability."""
        top = results_df.head(top_n)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(top)), top["pct_win"].values, color="steelblue")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["team"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Win Probability (%)")
        ax.set_title(f"Top {top_n} Favorites - FIFA World Cup 2026")

        for bar, pct in zip(bars, top["pct_win"].values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.2f}%", va="center", fontsize=9)

        plt.tight_layout()
        path = OUTPUTS_DIR / "bracket_simulation.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[OK] Plot saved to {path}")

    @staticmethod
    def print_summary(results_df: pd.DataFrame):
        """Print a console summary of results."""
        print("\n" + "=" * 60)
        print("  FIFA WORLD CUP 2026 - PREDICTION RESULTS")
        print("=" * 60)

        champion = results_df.iloc[0]
        print(f"\n  Predicted Champion: {champion['team']}")
        print(f"     Win probability: {champion['pct_win']:.2f}%")

        print(f"\n  Top 10 Contenders:")
        print(f"  {'Rank':<6} {'Team':<20} {'Win%':<8} {'Final%':<8} {'SF%':<8} {'QF%':<8}")
        print(f"  {'-'*56}")
        for i, (_, row) in enumerate(results_df.head(10).iterrows()):
            print(f"  {i+1:<6} {row['team']:<20} "
                  f"{row['pct_win']:<8.2f} {row['pct_final']:<8.2f} "
                  f"{row['pct_sf']:<8.2f} {row['pct_qf']:<8.2f}")


if __name__ == "__main__":
    model = load_model()
    config = load_group_config()
    sim = TournamentSimulator(model, config)
    results = sim.run(n_simulations=100)
    sim.print_summary(results)
    sim.plot_results(results)

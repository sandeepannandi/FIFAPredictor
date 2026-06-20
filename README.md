# FIFA World Cup 2026 Winner Prediction

A machine learning system that predicts the winner of the FIFA World Cup 2026 by training a Random Forest classifier on 74 years of international football results and running a Monte Carlo simulation of the expanded 48-team tournament.

---

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Outputs](#outputs)
- [2026 World Cup Format](#2026-world-cup-format)
- [Technical Details](#technical-details)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Overview

This project predicts the FIFA World Cup 2026 champion using a data-driven approach:

1. **Data Collection** -- Downloads the [International Football Results dataset](https://github.com/martj42/international_results) (1872-2024), containing over 50,000 international matches from all FIFA-recognised nations.
2. **Feature Engineering** -- Computes rolling ELO ratings, recent form indicators, goal-scoring averages, head-to-head records, and tournament importance weights for every match.
3. **Model Training** -- Trains a `RandomForestClassifier` (300 trees, max depth 8) with a time-based train/test split to prevent look-ahead bias.
4. **Monte Carlo Simulation** -- Simulates the full 2026 tournament 10,000 times, sampling match outcomes from the model's probability distribution and tracking each team's advancement rate.

---

## Methodology

### Feature Engineering

Each match in the training set is transformed into a 14-dimensional feature vector computed exclusively from data available *before* the match date:

| Feature | Description |
|---------|-------------|
| ELO rating difference | Difference in computed ELO ratings between the two teams |
| Home form | Average points per match from the last 10 matches (3 for win, 1 for draw, 0 for loss) |
| Away form | Same for the opponent |
| Goals scored average | Average goals scored in the last 10 matches (per team) |
| Goals conceded average | Average goals conceded in the last 10 matches (per team) |
| Head-to-head win rate | Win percentage in the last 10 meetings between the two teams |
| Venue flags | Binary indicators for home or neutral venue |
| Tournament weight | Importance weight (1-4) based on competition type (friendly to World Cup) |
| Days since last match | Days elapsed since each team's previous match (fatigue proxy) |

### Model

A `RandomForestClassifier` is configured with:
- 300 decision trees
- Maximum depth of 8 to prevent overfitting
- Balanced class weights to handle the natural class imbalance (wins are more common than draws)
- Time-based 85/15 train/test split (no random shuffling)

### Tournament Simulation

The 2026 World Cup features an expanded format:
- **48 teams** split into **12 groups of 4**
- Top 2 from each group advance automatically (24 teams)
- The 8 best third-placed teams also advance
- 32-team knockout bracket: Round of 32 -> Round of 16 -> Quarter-Finals -> Semi-Finals -> Final

For each group match, the model produces Win/Draw/Loss probabilities. A Monte Carlo sample determines the result, and goal counts are sampled from a Poisson distribution conditioned on the outcome. Group standings are determined by points, then goal difference, then goals scored, then team strength.

Knockout matches use normalised win probabilities (draws are redistributed proportionally) to determine the winner, simulating the effect of extra time and penalty shootouts.

---

## Project Structure

```
fifa-2026-predictor/
├── data/
│   ├── raw/                      # Original CSV datasets (downloaded automatically)
│   ├── processed/                # Cleaned parquet files and feature matrices
│   └── fixtures/                 # 2026 groups and qualified teams configuration
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Download and clean raw match data
│   ├── feature_engineering.py    # Compute rolling features and ELO ratings
│   ├── train_model.py            # Train and evaluate RandomForest classifier
│   ├── simulate.py               # Monte Carlo tournament simulator
│   └── utils.py                  # ELO computation, constants, helpers
├── models/
│   └── rf_model.pkl              # Trained RandomForest model (joblib)
├── outputs/
│   ├── win_probabilities.csv     # Per-team advancement probabilities
│   ├── bracket_simulation.png    # Bar chart of top favorites
│   └── calibration_curve.png     # Model calibration visualisation
├── notebooks/
│   └── eda.ipynb                 # (optional) Exploratory data analysis
├── requirements.txt
├── main.py                       # CLI entry point
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11 or later
- pip (Python package manager)

### Installation

```bash
# Navigate to the project directory
cd fifa-2026-predictor

# Install Python dependencies
pip install -r requirements.txt
```

The dependencies are:
- `pandas` and `numpy` for data manipulation
- `scikit-learn` for the RandomForest classifier
- `matplotlib` and `seaborn` for visualisation
- `joblib` for model serialisation
- `requests` for dataset download

---

## Usage

### Quick Start

Run the full pipeline end-to-end with a single command:

```bash
python main.py
```

This will:
1. Download the international football results dataset (first run only, ~1 minute)
2. Clean and process the data, compute ELO ratings and feature vectors
3. Train and evaluate the RandomForest model
4. Run 10,000 tournament simulations
5. Display the predicted champion and top contenders
6. Save results to the `outputs/` directory

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--steps` | Comma-separated steps to run: `load`, `train`, `simulate`, or `all` | `all` |
| `--simulations` | Number of Monte Carlo simulations | 10000 |
| `--force-rebuild` | Force re-download and reprocess all data | (flag) |
| `--top-n` | Number of top teams to display | 10 |

### Examples

```bash
# Faster run with fewer simulations for testing
python main.py --simulations 1000

# Rerun only the simulation step (keep existing model)
python main.py --steps simulate

# Full rebuild from scratch
python main.py --force-rebuild

# Show more contenders
python main.py --top-n 20
```

---

## Outputs

After running the pipeline, the following files are produced in the `outputs/` directory:

| File | Description |
|------|-------------|
| `win_probabilities.csv` | Per-team probabilities for each knockout round (R32 through champion) |
| `bracket_simulation.png` | Horizontal bar chart showing the top N teams by win probability |
| `calibration_curve.png` | Three-panel calibration plot showing model reliability for Win/Draw/Loss |

The `win_probabilities.csv` file contains the following columns:
- `team` -- National team name
- `pct_r32` -- Percentage of simulations in which the team reached the Round of 32
- `pct_r16` -- Percentage reaching the Round of 16
- `pct_qf` -- Percentage reaching the Quarter-Finals
- `pct_sf` -- Percentage reaching the Semi-Finals
- `pct_final` -- Percentage reaching the Final
- `pct_win` -- Percentage of simulations won (primary prediction metric)

---

## 2026 World Cup Format

### Qualified Teams

The 48 qualified nations (as of March 2026) span all six confederations, including debutants Cape Verde, Curacao, Jordan, and Uzbekistan.

### Group Stage

| Group | Teams |
|-------|-------|
| A | Mexico, South Africa, South Korea, Czech Republic |
| B | Canada, Bosnia and Herzegovina, Qatar, Switzerland |
| C | Brazil, Morocco, Haiti, Scotland |
| D | USA, Paraguay, Australia, Turkey |
| E | Germany, Curacao, Ivory Coast, Ecuador |
| F | Netherlands, Japan, Sweden, Tunisia |
| G | Belgium, Egypt, Iran, New Zealand |
| H | Spain, Cape Verde, Saudi Arabia, Uruguay |
| I | France, Senegal, Iraq, Norway |
| J | Argentina, Algeria, Austria, Jordan |
| K | Portugal, DR Congo, Uzbekistan, Colombia |
| L | England, Croatia, Ghana, Panama |

### Advancement Rules

- The top 2 teams from each group advance to the knockout stage (24 teams)
- The 8 best third-placed teams across all groups also advance
- Third-placed teams are ranked by: points, goal difference, goals scored, team strength proxy
- The knockout stage is single elimination: Round of 32, Round of 16, Quarter-Finals, Semi-Finals, and Final

---

## Technical Details

### Data Source

The project uses the [International Football Results from 1872 to Present](https://github.com/martj42/international_results) dataset maintained by Mart JuriSOO. This dataset is downloaded automatically from GitHub on first run. Matches before 1950 are excluded to ensure stable ELO ratings.

### ELO Rating System

A custom ELO rating system is implemented for each team with:
- Initial rating: 1500
- K-factor: 32 (adjusted by goal difference margin)
- Home advantage: 100 ELO points (not applied for neutral venues)
- Ratings are updated iteratively match-by-match in chronological order

### Random Seed

All stochastic processes are seeded with 42 for fully reproducible results.

---

## Reproducibility

The project is designed to produce identical results across runs when given the same input data:
- Random seed is set to 42 at the start of every run
- The RandomForest classifier uses `random_state=42`
- Monte Carlo simulations use NumPy's seeded random number generator

---

## License

Apache 2.0

---

*Built with Python, scikit-learn, and 74 years of international football history.*

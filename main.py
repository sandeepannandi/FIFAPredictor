#!/usr/bin/env python3
"""
main.py -- FIFA World Cup 2026 Winner Prediction Pipeline.

Run end-to-end:
    python main.py

Run specific steps:
    python main.py --steps load,train,simulate
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import download_results, load_results
from src.feature_engineering import build_features
from src.train_model import train, load_model
from src.simulate import TournamentSimulator
from src.data_loader import load_group_config
from src.utils import set_random_seed, OUTPUTS_DIR, DATA_PROCESSED


def main():
    parser = argparse.ArgumentParser(
        description="FIFA World Cup 2026 Winner Prediction Model"
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated: load,train,simulate (default: all)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10,000)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force re-download and re-process data",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top teams to display (default: 10)",
    )
    args = parser.parse_args()

    set_random_seed(42)

    steps = [s.strip() for s in args.steps.split(",")]

    if "all" in steps or "load" in steps or args.force_rebuild:
        print("\n" + "=" * 50)
        print("  STEP 1: Load and clean data")
        print("=" * 50)

        download_results()
        matches = load_results()

        print("\n" + "=" * 50)
        print("  STEP 2: Feature engineering")
        print("=" * 50)

        features = build_features(matches, cache=not args.force_rebuild)
        print(f"    Features: {len(features):,} rows")

    if "all" in steps or "train" in steps:
        print("\n" + "=" * 50)
        print("  STEP 3: Train model")
        print("=" * 50)

        model = train(force_rebuild=args.force_rebuild,
                      skip_if_exists=not args.force_rebuild)

    if "all" in steps or "simulate" in steps:
        print("\n" + "=" * 50)
        print("  STEP 4: Tournament simulation")
        print("=" * 50)

        print("[..] Loading model and group configuration...")
        model = load_model()
        group_config = load_group_config()

        print(f"[..] Initializing simulator with "
              f"{len(group_config['groups'])} groups...")
        total_teams = sum(len(v) for v in group_config["groups"].values())
        print(f"    Total teams: {total_teams}")

        # Load final ELO ratings from processed data
        elo_path = DATA_PROCESSED / "final_elo.parquet"
        elo_ratings = {}
        if elo_path.exists():
            print(f"[..] Loading realism ELO ratings from {elo_path.name}...")
            import pandas as pd
            elo_df = pd.read_parquet(elo_path)
            elo_ratings = dict(zip(elo_df["team"], elo_df["elo_rating"]))
        else:
            print("[!] Warning: final_elo.parquet NOT found. Using default ratings.")

        simulator = TournamentSimulator(model, group_config, elo_ratings=elo_ratings)
        results = simulator.run(n_simulations=args.simulations)

        print(f"\n[OK] Simulations complete. Generating visualizations...")
        simulator.plot_results(results, top_n=args.top_n)
        simulator.print_summary(results)

        print(f"\n{'='*50}")
        print("  PIPELINE COMPLETE")
        print(f"{'='*50}")
        print("  Results:")
        print(f"    Win probabilities:   {OUTPUTS_DIR / 'win_probabilities.csv'}")
        print(f"    Bracket chart:       {OUTPUTS_DIR / 'bracket_simulation.png'}")
        print(f"    Calibration curve:   {OUTPUTS_DIR / 'calibration_curve.png'}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()

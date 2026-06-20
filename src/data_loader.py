"""
data_loader.py -- Load and clean raw football match data.

Downloads the martj42 "International Football Results" dataset from GitHub
if not already present, and provides loading utilities for FIFA rankings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import io
import os
import json

from src.utils import (
    DATA_RAW, DATA_PROCESSED, DATA_FIXTURES, normalize_team_name
)

# -- URLs -----------------------------------------------------------------
RESULTS_URL = (
    "https://github.com/martj42/international_results/archive/refs/heads/master.zip"
)
RESULTS_CSV = "results.csv"
SHOOTOUTS_CSV = "shootouts.csv"

# -- Local paths ----------------------------------------------------------
LOCAL_RESULTS = DATA_RAW / RESULTS_CSV
LOCAL_SHOOTOUTS = DATA_RAW / SHOOTOUTS_CSV
FIFA_RANKINGS_URL = (
    "https://raw.githubusercontent.com/jalapic/engsoccerdata/master/"
    "data-fifa/ranking.csv"
)
LOCAL_FIFA_RANKINGS = DATA_RAW / "fifa_ranking.csv"
PROCESSED_MATCHES = DATA_PROCESSED / "matches_clean.parquet"


def download_results():
    """Download the martj42 dataset from GitHub if not present."""
    if LOCAL_RESULTS.exists():
        print("[OK] Results CSV already cached.")
        return

    print("[..] Downloading international football results dataset...")
    resp = requests.get(RESULTS_URL, timeout=60)
    resp.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    prefix = None
    for name in z.namelist():
        if name.endswith(".csv"):
            prefix = name[: name.find("/") + 1]
            break

    if prefix is None:
        raise RuntimeError("Could not find CSV files in the downloaded archive.")

    z.extract(f"{prefix}{RESULTS_CSV}", DATA_RAW)
    z.extract(f"{prefix}{SHOOTOUTS_CSV}", DATA_RAW)

    src_results = DATA_RAW / f"{prefix}{RESULTS_CSV}"
    src_shootouts = DATA_RAW / f"{prefix}{SHOOTOUTS_CSV}"
    if src_results.exists():
        src_results.rename(LOCAL_RESULTS)
    if src_shootouts.exists():
        src_shootouts.rename(LOCAL_SHOOTOUTS)

    extracted_dir = DATA_RAW / prefix.replace("/", "")
    if extracted_dir.exists():
        import shutil
        shutil.rmtree(extracted_dir, ignore_errors=True)

    print(f"[OK] Downloaded and extracted {len(z.namelist())} files.")


def load_results(cache: bool = True) -> pd.DataFrame:
    """
    Load and clean the results dataset.
    If cache=True and processed file exists, loads cached parquet instead.
    """
    if cache and PROCESSED_MATCHES.exists():
        print("[OK] Loading cached clean matches...")
        return pd.read_parquet(PROCESSED_MATCHES)

    download_results()

    print("[..] Loading and cleaning raw results...")
    df = pd.read_csv(LOCAL_RESULTS, dtype=object, low_memory=False)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
    df["neutral"] = df["neutral"].fillna(False).astype(bool)

    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])

    # Filter to relevant time range (ELO stabilisation after 1950)
    df = df[df["date"] >= "1950-01-01"]
    df = df[df["date"] <= "2024-12-31"]

    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)

    df = df[df["home_score"] >= 0]
    df = df[df["away_score"] >= 0]

    df = df.sort_values("date").reset_index(drop=True)

    print(f"[OK] Cleaned data: {len(df):,} matches, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"     Tournaments: {df['tournament'].nunique()} unique")

    df.to_parquet(PROCESSED_MATCHES, index=False)
    return df


def download_fifa_rankings():
    """Download FIFA World Ranking historical data."""
    if LOCAL_FIFA_RANKINGS.exists():
        print("[OK] FIFA rankings already cached.")
        return

    print("[..] Downloading FIFA rankings...")
    try:
        resp = requests.get(FIFA_RANKINGS_URL, timeout=30)
        resp.raise_for_status()
        with open(LOCAL_FIFA_RANKINGS, "wb") as f:
            f.write(resp.content)
        print(f"[OK] FIFA rankings saved ({len(resp.content):,} bytes)")
    except Exception as e:
        print(f"[!] Could not download FIFA rankings: {e}")
        print("    The model will use ELO ratings instead.")


def load_fifa_rankings() -> pd.DataFrame | None:
    """Load FIFA rankings if available."""
    if not LOCAL_FIFA_RANKINGS.exists():
        return None
    try:
        df = pd.read_csv(LOCAL_FIFA_RANKINGS)
        df["rank_date"] = pd.to_datetime(df.get("rank_date", df.get("date")), errors="coerce")
        df = df.dropna(subset=["rank_date"])
        return df
    except Exception as e:
        print(f"[!] Error loading FIFA rankings: {e}")
        return None


def load_group_config() -> dict:
    """Load the 2026 World Cup group configuration."""
    path = DATA_FIXTURES / "groups_2026.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Group config not found at {path}. "
            "Ensure data/fixtures/groups_2026.json exists."
        )
    with open(path) as f:
        return json.load(f)


def load_qualified_teams() -> list[str]:
    """Load the list of qualified teams and return a flat list."""
    path = DATA_FIXTURES / "qualified_teams.json"
    with open(path) as f:
        data = json.load(f)
    teams = []
    for conf_teams in data.values():
        teams.extend(conf_teams)
    return teams

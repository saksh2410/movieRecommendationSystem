"""
Configuration Module for the project.
This module stores the global parameters such as data paths, feature selection configurations, model hyperparameters used throughout the project.
"""

# imports
from pathlib import Path


# raw data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
MOVIES = RAW_DATA_PATH / "tmdb_5000_movies.csv"
CREDITS = RAW_DATA_PATH / "tmdb_5000_credits.csv"

# Feature selection attributes
KEEP_COLS = ['title','genres','keywords','overview', 'tagline', 'cast', 'crew']
# Reformatting attributes
CLEANUP_COLS = ['genres', 'keywords', 'cast']

# Model hyperparameters
TOP_N = 5

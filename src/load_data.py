# imports
import pandas as pd
from pathlib import Path

# raw data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
MOVIES = RAW_DATA_PATH / "tmdb_5000_movies.csv"
CREDITS = RAW_DATA_PATH / "tmdb_5000_credits.csv"

def load_movies(path=MOVIES):
    """
    Load the movies dataset from a CSV file.

    Parameters
    ----------
    path : str, optional
        File path to the movies CSV file, by default.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing movie metadata such as title, genre, overview.
    """
    return pd.read_csv(path)

def load_credits(path=CREDITS):
    """
    Load the credits dataset from a CSV file.

    Args
    ----------
    path : str, optional
        File path to the credits CSV file, by default "data\raw\tmdb_5000_movies.csv".

    Returns
    -------
    pandas.DataFrame
        DataFrame containing movie credits such as cast, crew.
    """
    return pd.read_csv(path)
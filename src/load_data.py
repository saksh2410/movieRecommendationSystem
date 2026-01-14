# imports
import pandas as pd
import config

def load_movies(path=config.MOVIES):
    """
    Load the movies dataset from a CSV file.

    Args
    ----------
    path : str, optional
        File path to the movies CSV file, by default.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing movie metadata such as title, genre, overview.
    """
    return pd.read_csv(path)

def load_credits(path=config.CREDITS):
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
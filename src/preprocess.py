# imports
import pandas as pd
import ast
from nltk.stem import PorterStemmer
import config


def merge_movies_credits(df_movies: pd.DataFrame, df_credits: pd.DataFrame):
    """
    function to merge the movies and credit dataframes on movie title and return a merged dataframe.

    Args
    ----------
    df_movies: pandas.Dataframe
        Dataframe with movie title, genre, overview and other details.
    df_credits: pandas.Dataframe
        Dataframe with movie title, cast, and crew.

    Returns
    ----------
    pandas.Dataframe
        Dataframe after merging
    """
    return pd.merge(df_movies, df_credits, on="title")


def clean_dataframe(df: pd.DataFrame):
    """
    function to perform feature selection, drop nulls, remove duplicates and clean up the dataframe.

    Args
    ----------
    df: pandas.Dataframe
        Dataframe to be cleaned

    Returns
    ----------
    pandas.Dataframe
        Cleaned up dataframe with only relevant columns
    """

    # keep only relevant columns
    df = df.loc[:, config.KEEP_COLS]

    # drop null or duplicated columns
    df = df.drop_duplicates()
    df = df.dropna()

    def format_column_values(obj: str):
        """
        function to clean up column values into proper format.
        it convert string values to list of strings and retain only relevant information

        Args
        ----------
        obj: str
            object to be cleaned up

        Returns
        ----------
        str
            cleaned up object
        """
        string_list = ast.literal_eval(obj)
        data = ""
        for item in string_list:
            # Formatting strings by stripping spaces in list attributes to avoid vectorizing them separately.
            # For example "Tom  Cruise" and "Tom Holland" should be separate single vectors without semantic similarities.
            data += item["name"].replace(" ", "") + " "
        return data

    # format data values for certain columns
    for col in config.CLEANUP_COLS:
        df[col] = df[col].apply(format_column_values)

    # Create a column for the director name from crew and drop the crew attribute.
    def fetch_director(crew: list[dict]):
        """
        This function fetches the name of the director from the entire crew list.

        Args:
            crew (list[dict]): a list of all crew members and details such as dept, gender, id etc.

        Returns:
            str: Name of the director
        """
        for crew_member in ast.literal_eval(crew):
            if crew_member["job"] == "Director":
                return crew_member["name"].replace(" ", "")
        return ""

    df["director"] = df["crew"].apply(fetch_director)
    df = df.drop("crew", axis=1)
    return df


def generate_tags(df: pd.DataFrame):
    """
    function to generate tags by combining relevant text columns.

    Args
    ----------
    df: pandas.Dataframe
        Dataframe to generate tags for

    Returns
    ----------
    pandas.Dataframe
        Dataframe with an additional 'tags' column
    """
    df_tagged = pd.DataFrame(
        {   "title": df["title"],
            "tags": (
                df["overview"]
                + " "
                + df["tagline"]
                + " "
                + df["genres"].apply(lambda x: " ".join(x))
                + " "
                + df["keywords"].apply(lambda x: " ".join(x))
                + " "
                + df["cast"].apply(lambda x: " ".join(x))
                + " "
                + df["director"]
            )
        }
    )
    return df_tagged


def stem_tags(df: pd.DataFrame):
    """
    function to perform stemming on the tags column.

    Args
    ----------
    df: pandas.Dataframe
        Dataframe to perform stemming on

    Returns
    ----------
    pandas.Dataframe
        Dataframe with stemmed tags
    """
    ps = PorterStemmer()

    def stem_text(text: str):
        """
        function to stem the input text.

        Args
        ----------
        text: str
            text to be stemmed

        Returns
        ----------
        str
            stemmed text
        """
        y = []
        for word in text.split():
            y.append(ps.stem(word))
        return " ".join(y)

    df["tags"] = df["tags"].apply(stem_text)
    return df

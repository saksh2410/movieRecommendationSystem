# imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config


def create_count_matrix(corpus):
    """
    Function to create a count matrix from the given corpus using CountVectorizer.

    Args
    ----------
    corpus: list of str
        List of text documents.

    Returns
    ----------
    numpy.ndarray
        Count matrix representing the frequency of terms in the corpus.
    """
    vectorizer = CountVectorizer(stop_words='english')
    count_matrix = vectorizer.fit_transform(corpus).toarray()
    return count_matrix

def compute_tfidf(matrix):
    """
    Function to compute the TF-IDF representation from the count matrix.

    Args
    ----------
    matrix: numpy.ndarray
        Count matrix representing the frequency of terms.

    Returns
    ----------
    numpy.ndarray
        TF-IDF matrix.
    """
    import numpy as np
    import math

    nrow = matrix.shape[0]
    idf = []
    for i in range(matrix.shape[1]):
        doc_count = sum(1 for row in range(nrow) if matrix[row][i] != 0)
        idf.append(doc_count)

    tfidf_matrix = np.zeros_like(matrix, dtype=float)
    for row in range(nrow):
        num_words = sum(val for val in matrix[row])
        for i in range(len(matrix[row])):
            tf = matrix[row][i] / num_words if num_words > 0 else 0
            idf_value = math.log10(nrow / (1 + idf[i])) if idf[i] > 0 else 0
            tfidf_matrix[row][i] = tf * idf_value

    return tfidf_matrix

def calculate_cosine_similarity(matrix):
    """
    Function to calculate the cosine similarity between rows of the given matrix.

    Args
    ----------
    matrix: numpy.ndarray
        Input matrix (e.g., TF-IDF matrix).

    Returns
    ----------
    numpy.ndarray
        Cosine similarity matrix.
    """
    similarity_matrix = cosine_similarity(matrix)
    return similarity_matrix

def get_recommendations(similarity_matrix, titles, target_title, top_n=config.TOP_N):
    """
    Function to get movie recommendations based on cosine similarity.

    Args
    ----------
    similarity_matrix: numpy.ndarray
        Cosine similarity matrix.
    titles: list of str
        List of movie titles corresponding to the rows/columns of the similarity matrix.
    target_title: str
        Title of the target movie for which recommendations are sought.
    top_n: int, optional
        Number of top recommendations to return, by default 10.

    Returns
    ----------
    list of tuples
        List of recommended movie titles and their similarity scores.
    """
    if target_title not in titles:
        raise ValueError("This movie is not in the list")

    target_index = titles.index(target_title)
    similarity_scores = list(enumerate(similarity_matrix[target_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    recommended_movies = [(titles[i], score) for i, score in similarity_scores[1:top_n+1]]
    return recommended_movies


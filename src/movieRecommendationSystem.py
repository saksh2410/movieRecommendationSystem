import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer
import ast
import math

# Loading relevant data into a dataframe
movies= pd.read_csv("tmdb_5000_movies.csv")
credits= pd.read_csv("tmdb_5000_credits.csv")
df= pd.merge(movies, credits, on= "title")
df= df[['title','genres','keywords','overview','production_companies', 'tagline', 'cast']]

# Defining a function which helps clean data entries into proper format.
    # The data in certain columns was given in the form of a dictionary wherein only the 'name' was relevant.
def formatter(str):
    stringList= ast.literal_eval(str)
    ans=[]
    for item in stringList:
        ans.append(item['name'])
    return ans

# Creating a list of documents where each record is  stored a single string document
trainingData= []
for row in range(df.shape[0]):
    df.at[row, 'genres']= formatter(df.loc[row]['genres'])
    df.at[row, 'cast']= formatter(df.loc[row]['cast'])
    df.at[row, 'keywords']= formatter(df.loc[row]['keywords'])
    df.at[row, 'production_companies']= formatter(df.loc[row]['production_companies'])
    trainingData.append(str(df.iloc[row]))

# creating a matrix of vectors for each of the document
vectorizer= CountVectorizer(stop_words= 'english')
vectorMatrix= (vectorizer.fit_transform(trainingData)).toarray()
nrow= vectorMatrix.shape[0]

def tfidf(matrix):
    idf= []
    for i in range(matrix.shape[1]):
        docCount=0
        for row in range(nrow):
            if matrix[row][i] != 0:
                docCount += 1
        idf.append(docCount)
    return idf

idfVector= tfidf(vectorMatrix)

for row in range(nrow):
    numWords= sum(val for val in vectorMatrix[row])
    for i in range(len(vectorMatrix[row])):
        vectorMatrix[row][i] /= numWords
        vectorMatrix[row][i] *= (math.log10(nrow/ (1+ idfVector[i])))

#Adding the 'title' of the film to each vector in a dictionary to make them easy to call
finalMatrix= {}
for row in range(df.shape[0]):
    finalMatrix[df.iloc[row]['title']]= vectorMatrix[row]


movieList= finalMatrix.keys()
targetMovie= input("Enter Movie name:   ")
if targetMovie not in movieList:
    print("This movie is not in the list")
else:
    targetVector= finalMatrix.get(targetMovie, 0)
    cosineMatrix= {}
    for movie in movieList:
        movieVector= finalMatrix.get(movie)
        cosineMatrix[movie]= np.dot(targetVector,movieVector)/(norm(targetVector) * norm(movieVector))

    cosineMatrix= sorted(cosineMatrix.items(), key= lambda x: x[1], reverse= True)
    print(cosineMatrix)



# coding: utf-8

# In[2]:

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.
    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.
    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    #pass
    #print(movies['genres'])
    listTokens = []
    for genre in movies['genres']:
        listTokens.append(tokenize_string(genre))
    
    movies['tokens'] = pd.Series(listTokens, index = movies.index)
    #print(movies['tokens'].tolist())
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i
    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    #pass
    #print(movies)
    vocab = {}
    
    listVocab = []
    for tok in movies['tokens']:
        for t in tok:
            listVocab.append(t)
    
    uniqueGenres = set(listVocab)
    #print(sorted(uniqueGenres))
    
    counter = 0
    uniqueGenres = sorted(uniqueGenres)
    #print(uniqueGenres)
    for genres in uniqueGenres:
        vocab[genres] = counter
        counter += 1
 
    #print("vocab ",vocab)
    """Dict containing number of unique docs containing term i"""
    uniqueDocs = {}
    for toks in movies['tokens']:
        flag = {}
        for t in toks:
            flag[t] = True
        for t in toks:
            if t in uniqueDocs.keys() and flag:
                uniqueDocs[t] += 1
            else:
                uniqueDocs[t] = 1
                flag[t] = False
    
   
    numMovies = len(movies)
    num_features = len(vocab)
    listFeatures = []
    for toks in movies['tokens']:
        """dict containing token freqs for each movie; tf(i,d) and max_k tf(k,d)"""
        tokenFreq = {}
        for t in toks:
            if t in tokenFreq.keys():
                tokenFreq[t] += 1
            else:
                tokenFreq[t] = 1
        row = []
        column = []
        data = []
        maxFreq = tokenFreq[max(tokenFreq)]
        for key, values in tokenFreq.items():
            #print(values, maxFreq)
            temp1 = values/maxFreq
            temp2 = np.log10(numMovies/uniqueDocs[key])
            data.append(temp1 * temp2)
            row.append(0)
            #print(key, vocab[key])
            column.append(vocab[key])
        listFeatures.append(csr_matrix((data, (row, column)), shape=(1, num_features)).toarray())
    
    movies['features'] = pd.Series(listFeatures, index = movies.index)
    #print(movies['features'])
    return movies, vocab
        
    


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    #pass
    
    a = np.array(a)
    b = np.array(b)
    temp = a.dot(b.T)
    #print("dot prod",temp)
    
    normA = 0.0
    for x in a[0]:
        #print("X",x)
        normA += x*x
    norm1 = np.sqrt(normA)

    normB = 0.00
    
    for y in b[0]:
        #print("Y", y)
        normB += y*y
    norm2 = np.sqrt(normB)
    
    
    norm = norm1 * norm2
    #print(norm)
    cosine_sim = temp[0][0]/norm
    #print(cosine_sim)
    return cosine_sim


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.
    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.
    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.
    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    #pass
    #print(ratings_test)
   
    predList = []
    #print(movies['movieId'],movies['features'])
    #for userTest in ratings_test['userId']:
    for i in range(len(ratings_test)):
        movieIdTest = ratings_test['movieId'].iloc[i]
        userTest = ratings_test['userId'].iloc[i]
        ratings = ratings_train[ratings_train.userId == userTest]
        movieIdTrain = ratings_train[ratings_train.userId == userTest].movieId
        
        wSum = 0.00
        wValue = 0.00
        for movieB in movieIdTrain:
            A = movies[movies.movieId == movieIdTest].features.iloc[0]
            B = movies[movies.movieId == movieB].features.iloc[0]
            rating = ratings[ratings_train.movieId == movieB].rating.iloc[0]
            #print(rating)    
            
            weighted = cosine_sim(A, B)
                
            if(weighted > 0.00):
                wValue += weighted * rating
                wSum += weighted
            else:
                wValue += 0.00
                wSum += 0.00
        
        if(wSum <= 0.00):
            wAvg = sum(ratings.rating)/len(ratings)
            #print("mean rating avg",wAvg)
        else:
            wAvg = wValue/wSum
        predList.append(wAvg)
        
    #print(len(predList))
    predArray = np.array(predList)
    return predArray
                
def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()


# In[ ]:




# In[ ]:




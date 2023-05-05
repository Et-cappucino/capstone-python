#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from flask import Flask, request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')


# In[2]:


actor = pd.read_json('movieAppData/actor.json')
cast = pd.read_json('movieAppData/cast.json')
genres = pd.read_json('movieAppData/genres.json')
watchable = pd.read_json('movieAppData/watchable.json')


# ## Content-based model

# In[3]:


#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')


# In[4]:


#Replace NaN with an empty string
watchable['description'] = watchable['description'].fillna('')


# In[5]:


#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(watchable['description'])


# In[6]:


# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[7]:


indices = pd.Series(watchable.index, index=watchable['name']).drop_duplicates()


# In[8]:


idx = indices["Toy Story"]


# In[9]:


sim_scores = list(enumerate(cosine_sim[idx]))


# In[10]:


sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)


# In[11]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return watchable['name'].iloc[movie_indices]


# In[12]:


sim_scores = sim_scores[1:11]


# In[13]:


movie_indices = [i[0] for i in sim_scores]


# ## Advanced Content-based model

# In[14]:


genres = genres.groupby("watchable_id").agg({'genre': list}).reset_index()


# In[15]:


watchable = watchable.merge(genres, how = 'left', left_on='id', right_on= 'watchable_id')


# In[16]:


cast = cast.merge(actor[['id', "first_name", "last_name"]],how = 'left', left_on = 'actor_id', right_on = 'id')


# In[17]:


cast['actor name'] = cast['first_name'] + " " + cast['last_name']


# In[18]:


cast_new = cast.groupby("watchable_id").agg({"actor name": list}).reset_index()


# In[19]:


watchable = watchable.merge(cast_new, left_on = 'id', right_on = 'watchable_id')


# In[20]:


# to change the amount of data for fastening the processing --> data = watchable.iloc[:5000]
def content_based_filtering(movie_title, data = watchable, n_recommendations=5):
    """
    Recommends movies based on their genres, actors, descriptions, and titles using cosine similarity.

    Parameters:
        - movie_title: str, title of the movie for which we want to recommend similar movies
        - data: pd.DataFrame, dataframe containing movie titles, genres, actors, and descriptions
        - n_recommendations: int, number of recommendations to return (default=5)

    Returns:
        - recommendations: pd.DataFrame, dataframe containing recommended movies
    """
    data['actor name'] = data['actor name'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    data['genre'] = data['genre'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Compute the tf-idf matrix for the movie descriptions
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['description'])

    # Compute the cosine similarity matrix between movies based on descriptions
    cosine_sim_matrix_desc = cosine_similarity(tfidf_matrix)

    # Compute the cosine similarity matrix between movies based on genres and actors
    genres_actors_matrix = data[['genre', 'actor name']]
    genres_actors_matrix = pd.get_dummies(genres_actors_matrix)
    cosine_sim_matrix_ga = cosine_similarity(genres_actors_matrix)

    # Find the index of the movie in the dataframe
    movie_index = data[data['name'] == movie_title].index[0]

    # Compute the weighted cosine similarity between the movie and all other movies
    sim_scores = (0.5 * cosine_sim_matrix_desc[movie_index]) + (0.5 * cosine_sim_matrix_ga[movie_index])

    # Sort the movies by their similarity score
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top recommendations
    top_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]

    # Get the recommended movies from the dataframe
    recommendations = data.iloc[top_indices]['id'].tolist()

    return recommendations

def get_content_based_result(movie_title):
    result = content_based_filtering(movie_title)
    return result


# ## Genre-based model

# In[21]:


genres = pd.read_json('movieAppData/genres.json')
watchable = pd.read_json('movieAppData/watchable.json')

def recommend_according_to_genre(genre, n_movies):
    genre_movies = genres[genres['genre'].isin(genre)]['watchable_id']
    watched_movies = watchable[watchable['id'].isin(genre_movies)].sort_values(by = ['rating', "vote_count"], ascending = [ False, False])
    top_n_movies = watched_movies.head(n_movies)
    top_n_movies.reset_index(drop = True, inplace = True)
    result = top_n_movies["id"].to_dict()
    return result 

def get_genre_based_result(genres):
    result = recommend_according_to_genre(genres, 5)
    to_arr = lambda result: [result[i] for i in result]
    return to_arr(result)


# ## API Calls and Endpoints 

# In[ ]:


app = Flask(__name__)

@app.route('/api/recommend/content_based', methods = ['POST'])
def recommend_content_based():    
    request_data = request.get_json()
    title = request_data['title']
    return get_content_based_result(title)

@app.route('/api/recommend/genre_based', methods = ['POST'])
def recommend_genre_based():    
    request_data = request.get_json()
    genres = request_data['genres']
    return get_genre_based_result(genres)

if __name__ == '__main__':
    app.run()


# In[ ]:





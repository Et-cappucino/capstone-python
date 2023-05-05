#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys
from flask import Flask, request


# In[2]:


genres = pd.read_json('movieAppData/genres.json')
watchable = pd.read_json('movieAppData/watchable.json')


# In[3]:


def recommend_according_to_genre(genre, n_movies):
    genre_movies = genres[genres['genre'].isin(genre)]['watchable_id']
    watched_movies = watchable[watchable['id'].isin(genre_movies)].sort_values(by = ['rating', "vote_count"], ascending = [ False, False])
    top_n_movies = watched_movies.head(n_movies)
    top_n_movies.reset_index(drop = True, inplace = True)
    result = top_n_movies["id"].to_dict()
    return result 

def get_result(genres):
    result = recommend_according_to_genre(genres, 7)
    to_arr = lambda result: [result[i] for i in result]
    return to_arr(result)


# In[ ]:


app = Flask(__name__)

@app.route('/api/recommend/genre_based', methods = ['POST'])
def recommend():    
    request_data = request.get_json()
    genres = request_data['genres']
    return get_result(genres)


if __name__ == '__main__':
    app.run()


# In[ ]:





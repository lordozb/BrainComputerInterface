
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[46]:


header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)
df_movie = pd.read_csv('ml-100k/u.item', sep='|',usecols=['id','movie'])


# In[4]:


n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))


# In[5]:


from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)


# In[8]:


train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


# In[9]:


from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


# In[10]:


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# In[11]:


item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


# In[12]:


item_prediction[3]


# In[13]:


user_prediction[4]


# In[63]:


import math
def return_movies(user):
    user_vector = user_prediction[user]
    to_return = np.zeros((1,n_users))
    max_prediction = 0
    for usr in user_prediction:
        prediction = user_vector.dot(usr)
        if(max_prediction < prediction):
            to_return = usr
            max_prediction = prediction
    to_recommend_videos = to_return.argsort()[-3:][::-1]
    videos_to_return = []
    
    for video in to_recommend_videos:
        videos_to_return.append((df_movie.iloc[[video]]).movie.tolist())
        
    return videos_to_return       
        


# In[64]:


print(return_movies(3))


# In[62]:





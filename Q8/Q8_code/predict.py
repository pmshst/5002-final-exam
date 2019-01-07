from movielens import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import sys
import time
import pickle
import os
import pandas as pd
user = []
item1 = []
item = []
rating = []
rating_test = []
module_path = os.path.dirname(__file__)

def contain(Genres, substr):

    if substr.lower() in Genres.lower():
        return 1
    return 0
item1 = pd.read_csv('/Users/zhaocai/PycharmProjects/5002final/Q8/movies.csv')
item1['unknown'] =  0
item1['action'] =  item1.apply(lambda x: contain(x.Genres, 'Action'), axis = 1)
item1['adventure'] =   item1.apply(lambda x: contain(x.Genres, 'Adventure'), axis = 1)
item1['animation'] =  item1.apply(lambda x: contain(x.Genres, 'Animation'), axis = 1)
item1['childrens'] =    item1.apply(lambda x: contain(x.Genres, 'Children'), axis = 1)
item1['comedy'] = item1.apply(lambda x: contain(x.Genres, 'Comedy'), axis = 1)
item1['crime'] =  item1.apply(lambda x: contain(x.Genres, 'Crime'), axis = 1)
item1['documentary'] =    item1.apply(lambda x: contain(x.Genres, 'Documentary'), axis = 1)
item1['drama'] =     item1.apply(lambda x: contain(x.Genres, 'Drama'), axis = 1)
item1['fantasy'] =    item1.apply(lambda x: contain(x.Genres, 'Fantasy'), axis = 1)
item1['film_noir'] =  item1.apply(lambda x: contain(x.Genres, 'Film-Noir'), axis = 1)
item1['horror'] =    item1.apply(lambda x: contain(x.Genres, 'Horror'), axis = 1)
item1['musical'] =  item1.apply(lambda x: contain(x.Genres, 'Musical'), axis = 1)
item1['mystery'] =  item1.apply(lambda x: contain(x.Genres, 'Mystery'), axis = 1)
item1['romance'] =  item1.apply(lambda x: contain(x.Genres, 'Romance'), axis = 1)
item1['sci_fi'] =     item1.apply(lambda x: contain(x.Genres, 'Sci-Fi'), axis = 1)
item1['thriller'] =  item1.apply(lambda x: contain(x.Genres, 'Thriller'), axis = 1)
item1['war'] =     item1.apply(lambda x: contain(x.Genres, 'War'), axis = 1)
item1['western'] =  item1.apply(lambda x: contain(x.Genres, 'Western'), axis = 1)

del item1['Genres']
item1.to_csv('/Users/zhaocai/PycharmProjects/5002final/Q8/Movie_Recommendation/items.csv', index =None)

# Load the movie lens dataset into arrays
d = Dataset()
d.load_users('/Users/zhaocai/PycharmProjects/5002final/Q8/users.csv', user) # same
d.load_items("/Users/zhaocai/PycharmProjects/5002final/Q8/Movie_Recommendation/items.csv", item) # movies.csv
d.load_ratings('/Users/zhaocai/PycharmProjects/5002final/Q8/rating_train.csv', rating)# rating_train.csv
#d.load_ratings_test('/Users/zhaocai/PycharmProjects/5002final/Q8/rating_test.csv', rating_test) # rating_test.csv

df_rating_test = pd.read_csv('/Users/zhaocai/PycharmProjects/5002final/Q8/rating_test.csv')

n_users = 6040
n_items = 3952

movie_genre = []

for movie in item:
    movie_genre.append([movie.unknown, movie.action, movie.adventure, movie.animation, movie.childrens, movie.comedy,
                        movie.crime, movie.documentary, movie.drama, movie.fantasy, movie.film_noir, movie.horror,
                        movie.musical, movie.mystery, movie.romance, movie.sci_fi, movie.thriller, movie.war, movie.western])


movie_genre = np.array(movie_genre)
cluster = KMeans(n_clusters=19)
cluster.fit_predict(movie_genre)
utility_copy = pickle.load( open("utility_matrix.pkl", "rb") )
y_true = []
y_pred = []

def getRating(uid, mid):
    if(mid > len(cluster.labels_)-1):
        return 2.555
    return utility_copy[uid-1][cluster.labels_[mid-1]-1]

df_rating_test['Rating'] = df_rating_test.apply(lambda x: getRating(x.UserID, x.MovieID), axis = 1)

del df_rating_test['timestamps']

df_rating_test.to_csv('Q8_output.csv',index=None)
'''
f = open('Q8_output.csv', 'w')
f.write('UserID,MovieID,Rating\n ')
for i in df_rating_test['UserID']:
    for j in df_rating_test['MovieID']:
        #if test[i][j] == 0:
        if(j > len(cluster.labels_)-1):
            f.write("%d, %d, %.4f\n" % (i, j, 2.555))
            continue

        f.write("%d, %d, %.4f\n" % (i, j, utility_copy[i-1][cluster.labels_[j-1]-1]))
            #y_true.append(test[i][j])
            #y_pred.append(utility_copy[i][cluster.labels_[j]-1])
f.close()
'''


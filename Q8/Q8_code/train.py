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
d.load_ratings_test('/Users/zhaocai/PycharmProjects/5002final/Q8/rating_test.csv', rating_test) # rating_test.csv

n_users = 6040
n_items = 3952
utility = np.zeros((n_users, n_items))
for r in rating:
    utility[r.user_id-1][r.item_id-1] = r.rating

# Finds the average rating for each user and stores it in the user's object
for i in range(n_users):
    rated = np.nonzero(utility[i])
    n = len(rated[0])
    if n != 0:
        user[i].avg_r = np.mean(utility[i][rated])
    else:
        user[i].avg_r = 0.

print (utility)
test = np.zeros((n_users, n_items))
for r in rating_test:
    test[r.user_id - 1][r.item_id - 1] = r.rating
movie_genre = []

for movie in item:
    movie_genre.append([movie.unknown, movie.action, movie.adventure, movie.animation, movie.childrens, movie.comedy,
                        movie.crime, movie.documentary, movie.drama, movie.fantasy, movie.film_noir, movie.horror,
                        movie.musical, movie.mystery, movie.romance, movie.sci_fi, movie.thriller, movie.war, movie.western])


movie_genre = np.array(movie_genre)
cluster = KMeans(n_clusters=19)
cluster.fit_predict(movie_genre)
utility_clustered = []
for i in range(0, n_users):
    average = np.zeros(19)
    tmp = []
    for m in range(0, 19):
        tmp.append([])
    for j in range(0, n_items):
        if(j> len(cluster.labels_)-1):
            break
        if utility[i][j] != 0:
            tmp[cluster.labels_[j] - 1].append(utility[i][j])
    for m in range(0, 19):
        if len(tmp[m]) != 0:
            average[m] = np.mean(tmp[m])
        else:
            average[m] = 0
    utility_clustered.append(average)

utility_clustered = np.array(utility_clustered)
for i in range(0, n_users):
    x = utility_clustered[i]
    user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)
def pcs(x, y):
    num = 0
    den1 = 0
    den2 = 0
    A = utility_clustered[x - 1]
    B = utility_clustered[y - 1]
    num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r) for a, b in zip(A, B) if a > 0 and b > 0)
    den1 = sum((a - user[x - 1].avg_r) ** 2 for a in A if a > 0)
    den2 = sum((b - user[y - 1].avg_r) ** 2 for b in B if b > 0)
    den = (den1 ** 0.5) * (den2 ** 0.5)
    if den == 0:
        return 0
    else:
        return num / den
pcs_matrix = np.zeros((n_users, n_users))
for i in range(0, n_users):
    for j in range(0, n_users):
        if i!=j:
            pcs_matrix[i][j] = pcs(i + 1, j + 1)
            sys.stdout.write("\rGenerating Similarity Matrix [%d:%d] = %f" % (i+1, j+1, pcs_matrix[i][j]))
            sys.stdout.flush()
            time.sleep(0.00005)
print ("\rGenerating Similarity Matrix [%d:%d] = %f" % (i+1, j+1, pcs_matrix[i][j]))

print (pcs_matrix)
# Guesses the ratings that user with id, user_id, might give to item with id, i_id.
# We will consider the top_n similar users to do this.
def norm():
    normalize = np.zeros((n_users, 19))
    for i in range(0, n_users):
        for j in range(0, 19):
            if utility_clustered[i][j] != 0:
                normalize[i][j] = utility_clustered[i][j] - user[i].avg_r
            else:
                normalize[i][j] = float('Inf')
    return normalize
def guess(user_id, i_id, top_n):
    similarity = []
    for i in range(0, n_users):
        if i+1 != user_id:
            similarity.append(pcs_matrix[user_id-1][i])
    temp = norm()
    temp = np.delete(temp, user_id-1, 0)
    top = [x for (y,x) in sorted(zip(similarity,temp), key=lambda pair: pair[0], reverse=True)]
    s = 0
    c = 0
    for i in range(0, top_n):
        if top[i][i_id-1] != float('Inf'):
            s += top[i][i_id-1]
            c += 1
    g = user[user_id-1].avg_r if c == 0 else s/float(c) + user[user_id-1].avg_r
    if g < 1.0:
        return 1.0
    elif g > 5.0:
        return 5.0
    else:
        return g


utility_copy = np.copy(utility_clustered)
for i in range(0, n_users):
    for j in range(0, 19):
        if utility_copy[i][j] == 0:
            sys.stdout.write("\rGuessing [User:Rating] = [%d:%d]" % (i, j))
            sys.stdout.flush()
            time.sleep(0.00005)
            utility_copy[i][j] = guess(i+1, j+1, 150)
print ("\rGuessing [User:Rating] = [%d:%d]" % (i, j))
pickle.dump( utility_copy, open("utility_matrix.pkl", "wb"))
# Predict ratings for u.test and find the mean squared error


utility_matrix = pickle.load( open("utility_matrix.pkl", "rb") )





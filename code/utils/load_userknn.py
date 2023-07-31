import pandas as pd
import numpy as np
import pickle

# For each user, search for the 20 similar users with UserKNN, and output a shuffle list
# of unseen movie, from the neighbors
def search_50_unseen_movies(target_user, user_knn, threshold = 20):
    unseen_movies = set()
    data = pd.read_csv("../data/dataset/ml_small_2018/splitting/0/subset_train_200.tsv", sep="\t",
                       header=None, names=['userId', 'movieId', 'rating'])
    movie = pd.read_csv("../data/dataset/ml_small_2018/movies.csv", sep=",", usecols=['movieId', 'title'])
    df_target_user = data.loc[data['userId'] == target_user, ['userId', 'movieId', 'rating']]
    user_neighs = user_knn.get_user_neighs(target_user, threshold)
    for user, value in user_neighs:
        df_user = data.loc[data['userId'] == user, ['userId', 'movieId', 'rating']]
        df_user = df_user[~df_user['movieId'].isin(df_target_user['movieId'])].sort_values('rating',ascending=False)
        if len(unseen_movies) < 50:
            # For each similar user retrieve the first 10
            unseen_movies.update(df_user['movieId'].iloc[:10])
        else:
            movie = movie[movie['movieId'].isin(unseen_movies)]
            return movie.iloc[:50].sample(frac=1, replace=False)

def search_50_unlisten_artist(target_user, user_knn, threshold = 20):
    unlisten_artists = set()
    data = pd.read_csv('../data/dataset/hetrec2011_lastfm_2k/splitting/0/train_with_name.tsv', sep="\t",
                        header=None, names=['userId', 'artistId', 'weight', 'name', 'url', 'pictureURL'],
                        usecols=['userId', 'artistId', 'weight', 'name'])
    artist = pd.read_csv('../data/dataset/hetrec2011_lastfm_2k/artists.dat', sep="\t", usecols=['id', 'name'])
    df_target_user = data.loc[data['userId'] == target_user, ['userId', 'artistId', 'weight']]
    user_neighs = user_knn.get_user_neighs(target_user, threshold)
    for user, value in user_neighs:
        df_user = data.loc[data['userId'] == user, ['userId', 'artistId', 'weight']]
        df_user = df_user[~df_user['artistId'].isin(df_target_user['artistId'])].sort_values('weight',ascending=False)
        if len(unlisten_artists) < 50:
            # For each similar user retrieve the first 10
            unlisten_artists.update(df_user['artistId'].iloc[:10])
        else:
            artist = artist[artist['id'].isin(unlisten_artists)]
            return artist.iloc[:50].sample(frac=1, replace=False)

def search_50_unread_book(target_user, user_knn, threshold = 20):
    unread_book = set()
    data = pd.read_csv('../data/dataset/facebook_book/trainingset_with_name.tsv', sep="\t",
                        header=None, names=['userId','bookId', 'rating', 'name'])
    book = pd.read_csv('../data/dataset/facebook_book/mappingLinkedData.tsv', sep="\t",
                         names=['bookId', 'resourceURI'])
    book['resourceURI'] = book['resourceURI'].apply(lambda x: x.split('/')[-1].replace('_', ' '))
    df_target_user = data.loc[data['userId'] == target_user, ['userId', 'bookId', 'rating']]
    user_neighs = user_knn.get_user_neighs(target_user, threshold)
    for user, value in user_neighs:
        df_user = data.loc[data['userId'] == user, ['userId', 'bookId', 'rating']]
        df_user = df_user[~df_user['bookId'].isin(df_target_user['bookId'])].sort_values('rating',ascending=False)
        if len(unread_book) < 50:
            # For each similar user retrieve the first 10
            unread_book.update(df_user['bookId'].iloc[:10])
        else:
            artist = book[book['bookId'].isin(unread_book)]
            return artist.iloc[:50].sample(frac=1, replace=False)

class UserKnnLoader:

    def __init__(self, path):
        self.load_weights(path)
        self.create_public()


    def create_public(self):
        self._public_users = {v: k for k, v in self._private_users.items()}

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._similarity = saving_dict['_similarity']
        self._num_neighbors = saving_dict['_num_neighbors']
        self._implicit = saving_dict['_implicit']
        self._private_users = saving_dict['_private_users']
        self._similarity_matrix = saving_dict['_similarity_matrix']
        #self._similarity_matrix = convert_indices(saving_dict['_private_users'], saving_dict['_similarity_matrix']) #HetRec
        #self._similarity_matrix = pd.DataFrame(saving_dict['_similarity_matrix'], index=saving_dict['_private_users'], columns=saving_dict['_private_users'])

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))
    def get_user_neighs(self, u, k):
        user_id = self._public_users.get(u)
        sim = self._similarity_matrix[user_id]
        sim[user_id] = -np.inf
        indices, values = zip(*[(self._private_users.get(u_list[0]), u_list[1])
                              for u_list in enumerate(sim)])

        # indices, values = zip(*predictions.items())
        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    def get_user_neighs_df(self, u, k):
        user_id = self._public_users.get(u)
        sim = self._similarity_matrix.iloc[user_id].values
        sim[user_id] = -np.inf
        indices, values = zip(*[(self._private_users.get(u_list[0]), u_list[1])
                                for u_list in enumerate(sim)])

        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]


def convert_indices(dictionary, matrix):
    # Create a pandas DataFrame with the matrix
    df = pd.DataFrame(matrix)
    # Rename the row and column indices using the dictionary
    df.rename(index=dictionary, columns=dictionary, inplace=True)
    # Convert the DataFrame back to a numpy array
    converted_matrix = df.to_numpy()
    return df


if __name__ == '__main__':
    path = '../../results/user_knn_models/user_knn_hetrec2011_lastfm_2k/' \
           'UserKNN_nn=100_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=/' \
           'best-weights-UserKNN_nn=100_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights='
    user_knn = UserKnnLoader(path)
    user_1_neighs = user_knn.get_user_neighs(2, 10)
    top50_unseen_movies = search_50_unlisten_artist(2, user_knn)
    pass
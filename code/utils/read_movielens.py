import pandas as pd

def merge_titles_with_movies(ratings_df, movies_df):
    # Merge the two DataFrames on the 'MovieID' column
    ratings_df = pd.merge(ratings_df, movies_df, how='left', on='movieId')
    ratings_df = ratings_df.drop(columns='movieId')
    return ratings_df


class ReadData:
    def __init__(self, path):
        self.path = path

    def read_ratings(self):
        # Read the file with pandas, specifying separator, column names, and dropping the 'timestamp' column
        movielens_ratings = pd.read_csv(self.path + 'ratings.csv', sep=',', header=0,
                                        names=['userId', 'movieId', 'rating', 'timestamp'],
                                        usecols=['userId', 'movieId', 'rating']
                                        )
        # Modify the data types of columns
        movielens_ratings['userId'] = movielens_ratings['userId'].astype('int32')
        movielens_ratings['movieId'] = movielens_ratings['movieId'].astype('int32')
        movielens_ratings['rating'] = movielens_ratings['rating'].astype('float32')

        # Print the first few rows of the dataframe to check if it was read correctly
        print(movielens_ratings.head())
        return movielens_ratings

    def read_movies(self):
        # Read the file with pandas, specifying separator, column names, and dropping the 'timestamp' column
        movielens_movies = pd.read_csv(self.path + 'movies.csv', sep=',', header=0,
                                       names=['movieId', 'title', 'genres'],
                                       usecols=['movieId', 'title'])
        # Modify the data types of columns
        movielens_movies['movieId'] = movielens_movies['movieId'].astype('int32')
        movielens_movies['title'] = movielens_movies['title'].astype('str')

        # Print the first few rows of the dataframe to check if it was read correctly
        print(movielens_movies.head())
        return movielens_movies
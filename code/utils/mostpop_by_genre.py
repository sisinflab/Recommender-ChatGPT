import pandas as pd
import typing as t

overall_set = set()

def get_movie(genre_list: t.List, threshold: int) -> int:
    acc = 0
    for movie in genre_list:
        if acc >= threshold:
            break
        if movie in overall_set:
            pass
        else:
            acc += 1
            yield movie


if __name__ == "__main__":

    # Create a sample dataframe
    df = pd.DataFrame(columns=["movie_id", "genre"])
    ratings = pd.read_csv('../data/dataset/ml_small_2018/splitting/0/subset_train_230.tsv', sep='\t', header=None,
                          names=['userId', 'movieId', 'rating'])

    with open('../data/processed_data/map.tsv', 'r') as file:
        for line in file:
            pattern = line.strip().split("\t")
            movie_id = pattern[0]
            for genre in pattern[1:]:
                # Append the new row to the dataframe
                df = df.append({'movie_id': movie_id, 'genre': genre}, ignore_index=True)

    threshold = 5
    list_dict = {}

    for genre in df["genre"].unique():
        genre_movies = df[df["genre"] == genre]["movie_id"].tolist()
        ordered_movies_by_pop = ratings[ratings["movieId"].isin(genre_movies)].groupby('movieId')["userId"].count().reset_index("movieId").sort_values(by="userId", ascending=False)["movieId"].tolist()
        list_dict[genre] = ordered_movies_by_pop[:threshold]

    # Select up to 50 unique movies, considering repetitions
    selected_movies = set()
    index = 0

    for genre, movies in list_dict.items():
        # generator_genre = get_movie(movies, threshold)
        for movie in get_movie(movies, threshold):
            selected_movies.add(movie)

    # Save the results to a CSV file
    top_movies_by_genre = pd.DataFrame(list(selected_movies)[:50], columns=['movieId'])
    movies = pd.read_csv('../data/dataset/ml_small_2018/movies.csv', sep=',', header=0,
                         names=['movieId', 'title', 'genres'], usecols=['movieId', 'title'])
    top_movies_by_genre = pd.merge(top_movies_by_genre, movies, how='left', on='movieId')
    top_movies_by_genre.to_csv("../data/processed_data/top_50_movies_by_genre_eq_dist.tsv", sep='\t', index=False)
import pandas as pd
import csv
from external.elliot.run import run_experiment

def map_genres():
    genres_set = set()
    path = '../../data/dataset/ml_small_2018/movies.csv'.replace("\\","/")
    data = pd.read_csv(path, sep=',')

    data["genres"].apply(lambda x: genres_set.update(x.split("|")))
    genres_dict = {genre: id for id, genre in enumerate(genres_set)}

    data["genres_str"] = data["genres"].apply(lambda x: "\t".join([str(genres_dict[el]) for el in x.split("|")]))

    #header = "movieId\tgenres_str\n"
    with open("map.tsv", "w") as f:
        #f.write(header)  # Write header to file
        for index, row in data[["movieId", "genres_str"]].iterrows():
            f.write(str(row["movieId"]) + "\t" + row["genres_str"] + "\n")

    header = "genres_id\tgenres_str\n"
    with open("../../data/dataset/ml_small_2018/processed_data/features.tsv", "w") as f:
        f.write(header)
        for genre, id in genres_dict.items():
            f.write(str(id) + "\t" + genre + "\n")

def no_rerank_performance():
    top_50_movies = pd.read_csv("../../data/dataset/ml_small_2018/processed_data/top_50_movies_by_genre_eq_dist.tsv", sep="\t")
    user_ratings = pd.read_csv("../../data/dataset/ml_small_2018/splitting/0/subset_train_200.tsv", sep="\t",
                               header=None, names=["userId", "movieId", "rating"])
    result = []
    for user in user_ratings['userId'].unique():
        for rank in range(1, 51):
            result.append((user, top_50_movies['movieId'][rank-1], rank))

    with open('../../data/dataset/ml_small_2018/chat_gpt_output/no_rerank_performance.tsv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in result:
            writer.writerow(row)

def no_rerank_performance_hetrec():
    top50 = pd.read_csv('../../data/dataset/hetrec2011_lastfm_2k/processed_data/top50_most_pop_artists.tsv', sep='\t')
    user_ratings = pd.read_csv('../../data/dataset/hetrec2011_lastfm_2k/splitting/0/train.tsv', sep="\t",
                               header=None, names=["userID", "artistID", "weight"])
    result = []
    for user in user_ratings['userID'].unique():
        for rank in range(1, 51):
            result.append((user, top50['artistID'][rank-1], rank))

    with open('../../data/dataset/hetrec2011_lastfm_2k/processed_data/no_rerank_performance.tsv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in result:
            writer.writerow(row)

def no_rerank_performance_facebook():
    top50 = pd.read_csv('../../data/dataset/facebook_book/processed_data/top_50_books_by_genre.tsv', sep='\t')
    user_ratings = pd.read_csv('../../data/dataset/facebook_book/trainingset.tsv', sep="\t",
                               header=None, names=["userID", "bookId", "rating"])
    result = []
    for user in user_ratings['userID'].unique():
        for rank in range(1, 51):
            result.append((user, top50['bookId'][rank-1], rank))

    with open('../../data/dataset/facebook_book/chat_gpt_output/output_exp_2_rec_book_no_rerank_mostpop.tsv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in result:
            writer.writerow(row)

if __name__ == "__main__":
    no_rerank_performance_facebook()
    #run_experiment("../../code/elliot_config_files/hetrec/baseline_config_exp_2_no_rank.yml")

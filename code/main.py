import json

import openai.error

from open_ai_model import OpenAI
from utils.read_movielens import merge_titles_with_movies
from utils.utils import *
from utils.subset_creator import *
from external.elliot.run import run_experiment
import time
import os
import warnings
warnings.filterwarnings("ignore", message="Setuptools is replacing distutils.")

"""
   MovieLens Small
"""
# Rec :: Exp-1
def send_message_to_chat_gpt(token):
    # Retrieve ratings and movies information
    ratings = pd.read_csv('../data/dataset/ml_small_2018/splitting/0/subset_train_230.tsv', sep='\t',
                          header=None, names=['userId','movieId', 'rating'])
    movies = pd.read_csv('../data/dataset/ml_small_2018/movies.csv', sep=',', header=0,
                         names=['movieId', 'title', 'genres'], usecols=['movieId', 'title'])

    # Create the ratings dataframe with the movies title
    ratings = merge_titles_with_movies(ratings, movies)

    # Initialize variable for the request
    utils = Utils(ratings)
    chat_gpt = OpenAI('gpt-3.5-turbo', token)

    checkpoint_dir = '../data/dataset/ml_small_2018/chat_gpt_output/checkpoints/'
    last_user_checkpoint = None

    max_user_id = -1
    highest_user_file = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoint.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoint.txt")])
                if user_id > max_user_id:
                    max_user_id = user_id
                    highest_user_file = entry.path

    #last_user_checkpoint_file = highest_user_file
    if os.path.exists(highest_user_file):
        last_user_checkpoint = highest_user_file

    for user in ratings['userId'].unique():
        # Skip users until the last checkpoint is reached
        if last_user_checkpoint is not None and user <= max_user_id:
            continue

        # Generate the message for ChatGPT
        message = utils.movies_rated_by_user(user)

        # Send message to ChatGPT
        response = chat_gpt.request(message)

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoint.txt')
        with open(checkpoint_file, 'w') as f:
            f.write(response['choices'][0]['message']['content'])
def convert_chat_gpt_response(output_path):
    checkpoint_dir = '../data/dataset/ml_small_2018/chat_gpt_output/checkpoints/'
    recommendations = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoint.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoint.txt")])
                print(user_id)
                with open(entry, 'r') as file:
                    response = file.read()
                # Create the dictionary of recommendations
                recommendations = parse_movie_recommendations(response, user_id, recommendations)

    result = search_movies(recommendations)
    save_result(result, output_path)
# Re-rank MostPop :: Exp-2
def send_message_to_chat_gpt_for_rerank(token):
    # Retrieve ratings and movies information
    ratings = pd.read_csv('../data/dataset/ml_small_2018/splitting/0/subset_train_200.tsv', sep='\t',
                          header=None, names=['userId','movieId', 'rating'])
    movies = pd.read_csv('../data/dataset/ml_small_2018/movies.csv', sep=',', header=0,
                         names=['movieId', 'title', 'genres'], usecols=['movieId', 'title'])

    # Create the ratings dataframe with the movies title
    ratings = merge_titles_with_movies(ratings, movies)

    # Initialize variable for the request
    utils = Utils(ratings)
    chat_gpt = OpenAI('gpt-3.5-turbo', token)

    checkpoint_dir = '../data/dataset/ml_small_2018/chat_gpt_output/checkpoints_rerank/'
    last_user_checkpoint = None

    max_user_id = -1
    highest_user_file = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoint_rerank.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoint_rerank.txt")])
                if user_id > max_user_id:
                    max_user_id = user_id
                    highest_user_file = entry.path

    #last_user_checkpoint_file = highest_user_file
    if os.path.exists(highest_user_file):
        last_user_checkpoint = highest_user_file

    for user in ratings['userId'].unique():
        # Skip users until the last checkpoint is reached
        if last_user_checkpoint is not None and user <= max_user_id:
            continue

        # Generate the message for ChatGPT
        message = utils.rerank_by_user_profile(user)

        # Send message to ChatGPT
        response = chat_gpt.request(message)

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoint_rerank.txt')
        with open(checkpoint_file, 'w') as f:
            f.write(response['choices'][0]['message']['content'])
def convert_chat_gpt_response_for_rerank(output_path):
    checkpoint_dir = '../data/dataset/ml_small_2018/chat_gpt_output/checkpoints_rerank/'
    recommendations = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("checkpoint_rerank.txt"):
                user_id = int(entry.name[len("user_"):-len("checkpoints_rerank.txt")])
                print(user_id)
                with open(entry, 'r') as file:
                    response = file.read()
                # Create the dictionary of recommendations
                recommendations = parse_movie_recommendations(response, user_id, recommendations)

    result = search_movies(recommendations)
    save_result(result, output_path)
# Re-rank UserKNN :: Exp - 3
def send_message_to_chat_gpt_for_rerank_with_userknn(token):
    # Retrieve ratings and movies information
    ratings = pd.read_csv('../data/dataset/ml_small_2018/splitting/0/subset_train_200.tsv', sep='\t',
                          header=None, names=['userId','movieId', 'rating'])
    movies = pd.read_csv('../data/dataset/ml_small_2018/movies.csv', sep=',', header=0,
                         names=['movieId', 'title', 'genres'], usecols=['movieId', 'title'])

    # Create the ratings dataframe with the movies title
    ratings = merge_titles_with_movies(ratings, movies)

    # Initialize variable for the request
    utils = Utils(ratings)
    chat_gpt = OpenAI('gpt-3.5-turbo', token)

    checkpoint_dir = '../data/dataset/ml_small_2018/chat_gpt_output/checkpoints_rerank_userknn/'
    last_user_checkpoint = None

    max_user_id = -1
    highest_user_file = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoints_rerank_userknn.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoints_rerank_userknn.txt")])
                if user_id > max_user_id:
                    max_user_id = user_id
                    highest_user_file = entry.path

    #last_user_checkpoint_file = highest_user_file
    if os.path.exists(highest_user_file):
        last_user_checkpoint = highest_user_file

    for user in ratings['userId'].unique():
        # Skip users until the last checkpoint is reached
        if last_user_checkpoint is not None and user <= max_user_id:
            continue

        result = []
        # Generate the message for ChatGPT
        message = utils.rerank_by_similar_user_profile(user, result)

        # Send message to ChatGPT
        response = chat_gpt.request(message)

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoints_rerank_userknn.txt')
        with open(checkpoint_file, 'w') as f:
            f.write(response['choices'][0]['message']['content'])
        with open('../data/dataset/ml_small_2018/chat_gpt_output/no_rerank_userknn_performance.tsv', 'a',
                  newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            for row in result:
                writer.writerow(row)
def convert_chat_gpt_response_for_rerank_with_userknn(output_path):
    checkpoint_dir = '../data/dataset/ml_small_2018/chat_gpt_output/checkpoints_rerank_userknn/'
    recommendations = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoints_rerank_userknn.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoints_rerank_userknn.txt")])
                #print(user_id)
                with open(entry, 'r') as file:
                    response = file.read()
                # Create the dictionary of recommendations
                recommendations = parse_movie_recommendations(response, user_id, recommendations)

    result = search_movies(recommendations)
    save_result(result, output_path)



"""
    HetRec
"""
def send_message_to_chat_gpt_hetrec(token):
    # Retrieve train data
    train = pd.read_csv('../data/dataset/hetrec2011_lastfm_2k/splitting/0/train_with_name.tsv', sep="\t",
                        header=None, names=['userId','artistId', 'weight', 'name', 'url', 'pictureURL'],
                        usecols=['userId','artistId', 'weight', 'name'])

    # Initialize variable for the request
    utils = Utils(train)
    chat_gpt = OpenAI('gpt-3.5-turbo', token)

    checkpoint_dir = '../data/dataset/hetrec2011_lastfm_2k/chat_gpt_output/checkpoints/'
    last_user_checkpoint = None

    max_user_id = -1
    highest_user_file = None

    #Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoint.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoint.txt")])
                if user_id > max_user_id:
                    max_user_id = user_id
                    highest_user_file = entry.path

    if os.path.exists(highest_user_file):
        last_user_checkpoint = highest_user_file

    for user in train['userId'].unique():
        # Skip users until the last checkpoint is reached
        if last_user_checkpoint is not None and user <= max_user_id:
            continue

        # Generate the message for ChatGPT
        message = utils.artists_listened_by_user(user)

        # Send message to ChatGPT
        response = chat_gpt.request(message)

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoint.txt')
        with open(checkpoint_file, 'w') as f:
            f.write(response['choices'][0]['message']['content'])
def convert_chat_gpt_response_hetrec():
    checkpoint_dir = '../data/dataset/hetrec2011_lastfm_2k/chat_gpt_output/checkpoints/'
    recommendations = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoint.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoint.txt")])
                print(user_id)
                with open(entry, 'r') as file:
                    response = file.read()
                # Create the dictionary of recommendations
                recommendations = parse_artist_recommendations(response, user_id, recommendations)

    result = search_artist(recommendations)
    save_artist_result(result)
# Re-rank MostPop :: Exp-2
def send_message_to_chat_gpt_hetrec_rerank(token):
    # Retrieve train data
    train = pd.read_csv('../data/dataset/hetrec2011_lastfm_2k/splitting/0/train_with_name.tsv', sep="\t",
                        header=None, names=['userId','artistId', 'weight', 'name', 'url', 'pictureURL'],
                        usecols=['userId','artistId', 'weight', 'name'])

    # Initialize variable for the request
    utils = Utils(train)
    chat_gpt = OpenAI('gpt-3.5-turbo', token)

    checkpoint_dir = '../data/dataset/hetrec2011_lastfm_2k/chat_gpt_output/checkpoints_rerank/'
    last_user_checkpoint = None

    max_user_id = -1
    highest_user_file = None

    #Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoint_rerank.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoint_rerank.txt")])
                if user_id > max_user_id:
                    max_user_id = user_id
                    highest_user_file = entry.path

    if os.path.exists(highest_user_file):
        last_user_checkpoint = highest_user_file

    for user in train['userId'].unique():
        # Skip users until the last checkpoint is reached
        if last_user_checkpoint is not None and user <= max_user_id:
            continue

        # Generate the message for ChatGPT
        message = utils.rerank_by_user_profile_hetrec(user)

        # Send message to ChatGPT
        response = chat_gpt.request(message)

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoint_rerank.txt')
        with open(checkpoint_file, 'w') as f:
            f.write(response['choices'][0]['message']['content'])
def convert_chat_gpt_response_hetrec_rerank():
    checkpoint_dir = '../data/dataset/hetrec2011_lastfm_2k/chat_gpt_output/checkpoints_rerank/'
    recommendations = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoint_rerank.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoint_rerank.txt")])
                print(user_id)
                with open(entry, 'r') as file:
                    response = file.read()
                # Create the dictionary of recommendations
                recommendations = parse_artist_recommendations(response, user_id, recommendations)

    result = search_artist(recommendations)
    save_artist_result(result)
# Re-rank UserKNN :: Exp - 3
def send_message_to_chat_gpt_hetrec_for_rerank_with_userknn(token):
    # Retrieve train data
    train = pd.read_csv('../data/dataset/hetrec2011_lastfm_2k/splitting/0/train_with_name.tsv', sep="\t",
                        header=None, names=['userId', 'artistId', 'weight', 'name', 'url', 'pictureURL'],
                        usecols=['userId', 'artistId', 'weight', 'name'])

    # Initialize variable for the request
    utils = Utils(train)
    chat_gpt = OpenAI('gpt-3.5-turbo', token)

    checkpoint_dir = '../data/dataset/hetrec2011_lastfm_2k/chat_gpt_output/checkpoints_rerank_userknn/'
    last_user_checkpoint = None

    max_user_id = -1
    highest_user_file = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoints_rerank_userknn.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoints_rerank_userknn.txt")])
                if user_id > max_user_id:
                    max_user_id = user_id
                    highest_user_file = entry.path

    if os.path.exists(highest_user_file):
        last_user_checkpoint = highest_user_file

    for user in train['userId'].unique():
        # Skip users until the last checkpoint is reached
        if last_user_checkpoint is not None and user <= max_user_id:
            continue

        result = []
        # Generate the message for ChatGPT
        message = utils.rerank_by_similar_user_profile_hetrec(user, result)

        # Send message to ChatGPT
        response = chat_gpt.request(message)

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoints_rerank_userknn.txt')
        with open(checkpoint_file, 'w') as f:
            f.write(response['choices'][0]['message']['content'])

        # Save top50 no rerank for each user
        with open('../data/dataset/hetrec2011_lastfm_2k/chat_gpt_output/output_exp_2_no_rerank_userknn_performance.tsv', 'a',
                  newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            for row in result:
                writer.writerow(row)
def convert_chat_gpt_hetrec_response_for_rerank_with_userknn(output_path):
    checkpoint_dir = '../data/dataset/hetrec2011_lastfm_2k/chat_gpt_output/checkpoints_rerank_userknn/'
    recommendations = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoints_rerank_userknn.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoints_rerank_userknn.txt")])
                #print(user_id)
                with open(entry, 'r') as file:
                    response = file.read()
                # Create the dictionary of recommendations
                recommendations = parse_artist_recommendations(response, user_id, recommendations)

    result = search_artist(recommendations)
    save_result(result, output_path)



"""
    Facebook Book
"""
def send_message_to_chat_gpt_facebook_book(path, names, usecols, token, m_type, checkpoint_dir):
    """
    Send message to chatGPT
    :param token: openAI token
    :param path: location of the trainingset_with_name.tsv
    :param names: the names of the columns
    :param usecols: which columns to use
    :param checkpoint_dir: the name of the directory for saving the checkpoints
    :param m_type: EXP_1 Recommendation, EXP_2 Re-rank MostPop, EXP_3 Re-rank UserKNN
    """

    # Retrieve train data
    train = pd.read_csv(path, sep="\t", header=None, names=names, usecols=usecols)
    # Initialize variable for the request
    utils = Utils(train)
    chat_gpt = OpenAI('gpt-3.5-turbo', token)
    last_user_checkpoint = None
    max_user_id = -1
    highest_user_file = None

    # Iterate over the directory entries and find the file with the highest userId
    # with os.scandir(checkpoint_dir) as entries:
    #     for entry in entries:
    #         if entry.name.startswith("user_") and entry.name.endswith("_checkpoint.txt"):
    #             user_id = int(entry.name[len("user_"):-len("_checkpoint.txt")])
    #             if user_id > max_user_id:
    #                 max_user_id = user_id
    #                 highest_user_file = entry.path
    #
    # if os.path.exists(highest_user_file):
    #     last_user_checkpoint = highest_user_file

    for user in train['userId'].unique():
        # Skip users until the last checkpoint is reached
        if last_user_checkpoint is not None and user <= max_user_id:
            continue

        message = ''
        # Generate the message for ChatGPT
        if m_type == 'EXP_1':
            message = utils.book_read_by_user(user)
        if m_type == 'EXP_2':
            message = utils.rerank_by_user_profile_facebook(user)
        if m_type == 'EXP_3':
            result = []
            # Generate the message for ChatGPT
            message = utils.rerank_by_similar_user_profile_facebook(user, result)

        # Send message to ChatGPT
        response = chat_gpt.request(message)

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoint.txt')
        with open(checkpoint_file, 'w') as f:
            f.write(response['choices'][0]['message']['content'])

        if m_type == 'EXP_3':
            # Save top50 no rerank for each user
            with open('../data/dataset/facebook_book/chat_gpt_output/output_exp_3_rec_book_no_rerank_userknn.tsv', 'a',
                      newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                for row in result:
                    writer.writerow(row)
# TO-DO: Convert Response General
def convert_chat_gpt_facebook_book(checkpoint_dir, output_path):
    recommendations = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoint.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoint.txt")])
                #print(user_id)
                with open(entry, 'r') as file:
                    response = file.read()
                # Create the dictionary of recommendations
                recommendations = parse_item_recommendations(response, user_id, recommendations)

    result = search_item(recommendations, items_df = pd.read_csv('../data/dataset/facebook_book/books.tsv',
                                                                 sep='\t', names=['id', 'name']))
    save_result(result, output_path)

def main():
    # Facebook Books experiments
    token = 'token from OpenAI'
    try:
        send_message_to_chat_gpt_facebook_book(path='../data/dataset/facebook_book/trainingset_with_name.tsv',
                                               names=['userId','bookId', 'rating', 'name'],
                                               usecols=['userId','bookId', 'rating', 'name'],
                                               token=token,
                                               m_type='EXP_1',
                                               checkpoint_dir='../data/dataset/facebook_book/chat_gpt_output/checkpoints_rerank_userknn')
    except openai.error.Timeout as e:
        print("Request time out: {}".format(e))
        time.sleep(20)
        main()
    except openai.error.RateLimitError as e:
        print("API rate limit exceeded: {}".format(e))
        time.sleep(20)
        main()
    except openai.error.APIConnectionError as e:
        print("API connection error: {}".format(e))
        time.sleep(20)
        main()
    except json.JSONDecodeError as e:
        print("JSONDecodeError: {}".format(e))
        time.sleep(20)
        main()
    except openai.error.APIError as e:
        print("HTTP code 502 from API: {}".format(e))
        time.sleep(20)
        main()

    convert_chat_gpt_facebook_book(checkpoint_dir='../data/dataset/facebook_book/chat_gpt_output/checkpoints_rerank_userknn',
                                    output_path='../data/dataset/facebook_book/chat_gpt_output/output_exp_3_rec_book_re_rerank_userknn.tsv')
    cold_users = get_cold_test('../data/dataset/ml_small_2018/splitting/0/subset_test_200.tsv',
                              '../data/dataset/ml_small_2018/splitting/0/test.tsv')
    cold_users.to_csv("../data/dataset/ml_small_2018/splitting/0/subset_test_200_cold_users.tsv", sep="\t", header=None, index=False)

    # Elliot Baseline
    #run_experiment('elliot_config_files/DaVinci/hetrec/baseline_config_exp_3_re_rank_userknn.yml')
    pass

if __name__ == '__main__':
    main()
    pass
from open_ai_model import OpenAI
from utils.read_movielens import merge_titles_with_movies
from utils.utils import *
from utils.subset_creator import *
#from external.elliot.run import run_experiment

import json
import openai.error
import time
import os
import warnings

#E-mail setup
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Google Palm
import vertexai
from vertexai.preview.language_models import TextGenerationModel


warnings.filterwarnings("ignore", message="Setuptools is replacing distutils.")

"""
    Google Palm
"""
def predict_large_language_model_sample(project_id: str, model_name: str, temperature: float, max_decode_steps: int,
                                        top_p: float, top_k: int, content: str, location: str = "us-central1",
                                        tuned_model_name: str = "",) :
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
      model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    print(f"Response from Model: {response.text}")
    return response.text


"""
    MovieLens Dataset - OK
"""
def send_message_for_movielens(ratings_path, movies_path, m_type, checkpoint_dir):

    # Retrieve ratings and movies information
    ratings = pd.read_csv(ratings_path, sep='\t',header=None, names=['userId', 'movieId', 'rating'])
    movies = pd.read_csv(movies_path, sep=',', header=0, names=['movieId', 'title', 'genres'], usecols=['movieId', 'title'])

    # Create the ratings dataframe with the movies title
    ratings = merge_titles_with_movies(ratings, movies)

    # Initialize variable for the request
    utils = Utils(ratings)
    last_user_checkpoint = None
    max_user_id = -1
    highest_user_file = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith(f'_checkpoint_{m_type}.txt'):
                user_id = int(entry.name[len("user_"):-len(f'_checkpoint_{m_type}.txt')])
                if user_id > max_user_id:
                    max_user_id = user_id
                    highest_user_file = entry.path

    if os.path.exists(highest_user_file):
        last_user_checkpoint = highest_user_file

    for user in ratings['userId'].unique():
        # Skip users until the last checkpoint is reached
        if last_user_checkpoint is not None and user <= max_user_id:
            continue

        message = ''
        # Generate the message for ChatGPT
        if m_type == 'EXP_1':
            message = "Given a user, as a Recommender System, please provide the top 50 recommendations. " + utils.movies_rated_by_user(user)
            print(message)
        if m_type == 'EXP_2':
            message = "Given a user, act like a Recommender System." + utils.rerank_by_user_profile(user)
            print(message)
        if m_type == 'EXP_3':
            result = []
            # Generate the message for ChatGPT
            message = "Given a user, act like a Recommender System." + utils.rerank_by_similar_user_profile(user, result)
            print(message)

        response = predict_large_language_model_sample("long-carving-387112", "text-bison@001", 0, 900, 0.8, 40, f'''{message}''', "us-central1")

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoint_{m_type}.txt')
        with open(checkpoint_file, 'w') as f:
            f.write(response)
            #print(response['choices'][0]['text'])

        # Save Top-50 no-rerank for each user
        if m_type == 'EXP_3':
            with open(f'../data/dataset/ml_small_2018/google_palm/results_{m_type}_no_rerank.tsv', 'a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                for row in result:
                    writer.writerow(row)


"""
    HetRec Dataset
"""
def send_message_for_hetrec(path, names, usecols, m_type, checkpoint_dir):

    # Retrieve train data
    train = pd.read_csv(path, sep="\t", header=None, names=names, usecols=usecols)

    # Initialize variable for the request
    utils = Utils(train)
    last_user_checkpoint = None
    max_user_id = -1
    highest_user_file = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith(f'_checkpoint_{m_type}.txt'):
                user_id = int(entry.name[len("user_"):-len(f'_checkpoint_{m_type}.txt')])
                if user_id > max_user_id:
                    max_user_id = user_id
                    highest_user_file = entry.path

    if os.path.exists(highest_user_file):
        last_user_checkpoint = highest_user_file

    for user in train['userId'].unique():
        # Skip users until the last checkpoint is reached
        if last_user_checkpoint is not None and user <= max_user_id:
            continue

        message = ''
        # Generate the message for ChatGPT
        if m_type == 'EXP_1':
            message = "Given a user, as a Recommender System, please provide the top 50 recommendations. " + utils.artists_listened_by_user(user)
            print(message)
        if m_type == 'EXP_2':
            message = "Given a user, act like a Recommender System." + utils.rerank_by_user_profile_hetrec(user)
            print(message)
        if m_type == 'EXP_3':
            result = []
            # Generate the message for ChatGPT
            message = "Given a user, act like a Recommender System." + utils.rerank_by_similar_user_profile_hetrec(user, result)
            print(message)

        # Send message
        response = predict_large_language_model_sample("long-carving-387112", "text-bison@001", 0, 900, 0.8, 40, f'''{message}''', "us-central1")

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoint_{m_type}.txt')
        with open(checkpoint_file, 'w') as f:
            f.write(response)
            print(response)

        # Save Top-50 no-rerank for each user
        if m_type == 'EXP_3':
            with open(f'../data/dataset/hetrec2011_lastfm_2k/google_palm/results_{m_type}_no_rerank.tsv', 'a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                for row in result:
                    writer.writerow(row)


"""
    Facebook Book Dataset
"""
def send_message_for_facebook_book(path, names, usecols, m_type, checkpoint_dir):

    # Retrieve train data
    train = pd.read_csv(path, sep="\t", header=None, names=names, usecols=usecols)

    # Initialize variable for the request
    utils = Utils(train)
    last_user_checkpoint = None
    max_user_id = -1
    highest_user_file = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith(f'_checkpoint_{m_type}.txt'):
                user_id = int(entry.name[len("user_"):-len(f'_checkpoint_{m_type}.txt')])
                if user_id > max_user_id:
                    max_user_id = user_id
                    highest_user_file = entry.path

    if os.path.exists(highest_user_file):
        last_user_checkpoint = highest_user_file

    for user in train['userId'].unique():
        # Skip users until the last checkpoint is reached
        if last_user_checkpoint is not None and user <= max_user_id:
            continue

        message = ''
        # Generate the message for ChatGPT
        if m_type == 'EXP_1':
            message = "Given a user, as a Recommender System, please provide the top 50 recommendations. " + utils.book_read_by_user(user)
            print(message)
        if m_type == 'EXP_2':
            message = "Given a user, act like a Recommender System." + utils.rerank_by_user_profile_facebook(user)
            print(message)
        if m_type == 'EXP_3':
            result = []
            # Generate the message for ChatGPT
            message = "Given a user, act like a Recommender System." + utils.rerank_by_similar_user_profile_facebook(user, result)
            print(message)

        # Send message
        response = predict_large_language_model_sample("long-carving-387112", "text-bison@001", 0, 900, 0.8, 40, f'''{message}''', "us-central1")

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoint_{m_type}.txt')
        with open(checkpoint_file, 'w') as f:
            f.write(response)
            print(response)

        # Save Top-50 no-rerank for each user
        if m_type == 'EXP_3':
            with open(f'../data/dataset/facebook_book/google_palm/results_{m_type}_no_rerank.tsv', 'a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                for row in result:
                    writer.writerow(row)


"""
    Convert checkpoints to Results TSV - OK
"""
def convert_results(checkpoint_dir, output_path, m_type, item_path):
    recommendations = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith(f"_checkpoint_{m_type}.txt"):
                user_id = int(entry.name[len("user_"):-len(f"_checkpoint_{m_type}.txt")])
                with open(entry, 'r') as file:
                    response = file.read()
                # Create the dictionary of recommendations
                recommendations = parse_item_recommendations(response, user_id, recommendations)
    if item_path == '../data/dataset/ml_small_2018/movies.csv':
        result = search_movies(recommendations)
    elif item_path == '../data/dataset/hetrec2011_lastfm_2k/artists.dat':
        result = search_artist(recommendations)
    else:
        result = search_item(recommendations, items_df = pd.read_csv(item_path, sep='\t', names=['id', 'name']))
    save_result(result, output_path)

# MovieLens experiments - OK
def movielens(model):
    for m_type in ['EXP_1', 'EXP_2', 'EXP_3']:
        try:
            send_message_for_movielens(ratings_path='../data/dataset/ml_small_2018/splitting/0/subset_train_200.tsv',
                                       movies_path='../data/dataset/ml_small_2018/movies.csv',
                                       m_type=m_type,
                                       checkpoint_dir=f'../data/dataset/ml_small_2018/google_palm/{m_type}/')

        except openai.error.Timeout as e:
            print("Request time out: {}".format(e))
            time.sleep(20)
            movielens(model)
        except openai.error.RateLimitError as e:
            print("API rate limit exceeded: {}".format(e))
            time.sleep(20)
            movielens(model)
        except openai.error.APIConnectionError as e:
            print("API connection error: {}".format(e))
            time.sleep(20)
            movielens(model)
        except json.JSONDecodeError as e:
            print("JSONDecodeError: {}".format(e))
            time.sleep(20)
            movielens(model)
        except openai.error.APIError as e:
            print("HTTP code 502 from API: {}".format(e))
            time.sleep(20)
            movielens(model)
        except Exception as e:
            # Handle other types of exceptions
            print("An error occurred:", e)

        convert_results(checkpoint_dir = f'../data/dataset/ml_small_2018/google_palm/{m_type}/',
                        output_path = f'../data/dataset/ml_small_2018/google_palm/output_{m_type}.tsv',
                        m_type = m_type,
                        item_path = '../data/dataset/ml_small_2018/movies.csv')
    pass

# HetRec experiments - OK
def hetrec(model):
    for m_type in ['EXP_3']: #, 'EXP_2', 'EXP_3']:
        try:
            send_message_for_hetrec(path='../data/dataset/hetrec2011_lastfm_2k/splitting/0/train_with_name.tsv',
                                    names=['userId','artistId', 'weight', 'name', 'url', 'pictureURL'],
                                    usecols=['userId','artistId', 'weight', 'name'],
                                    m_type=m_type,
                                    checkpoint_dir=f'../data/dataset/hetrec2011_lastfm_2k/google_palm/{m_type}/')
        except openai.error.Timeout as e:
            print("Request time out: {}".format(e))
            time.sleep(20)
            hetrec(model)
        except openai.error.RateLimitError as e:
            print("API rate limit exceeded: {}".format(e))
            time.sleep(20)
            hetrec(model)
        except openai.error.APIConnectionError as e:
            print("API connection error: {}".format(e))
            time.sleep(20)
            hetrec(model)
        except json.JSONDecodeError as e:
            print("JSONDecodeError: {}".format(e))
            time.sleep(20)
            hetrec(model)
        except openai.error.APIError as e:
            print("HTTP code 502 from API: {}".format(e))
            time.sleep(20)
            hetrec(model)
        except Exception as e:
            # Handle other types of exceptions
            print("An error occurred:", e)

        convert_results(checkpoint_dir=f'../data/dataset/hetrec2011_lastfm_2k/google_palm/{m_type}/',
                        output_path=f'../data/dataset/hetrec2011_lastfm_2k/google_palm/output_{m_type}.tsv',
                        m_type=m_type,
                        item_path = '../data/dataset/hetrec2011_lastfm_2k/artists.dat')
    pass

# Facebook Books experiments - OK
def facebook_book(model):
    for m_type in ['EXP_1', 'EXP_2', 'EXP_3']:
        try:
            send_message_for_facebook_book(path='../data/dataset/facebook_book/trainingset_with_name.tsv',
                                           names=['userId','bookId', 'rating', 'name'],
                                           usecols=['userId','bookId', 'rating', 'name'],
                                           m_type=m_type,
                                           checkpoint_dir=f'../data/dataset/facebook_book/google_palm/{m_type}/')

        except openai.error.Timeout as e:
            print("Request time out: {}".format(e))
            time.sleep(20)
            facebook_book(model)
        except openai.error.RateLimitError as e:
            print("API rate limit exceeded: {}".format(e))
            time.sleep(20)
            facebook_book(model)
        except openai.error.APIConnectionError as e:
            print("API connection error: {}".format(e))
            time.sleep(20)
            facebook_book(model)
        except json.JSONDecodeError as e:
            print("JSONDecodeError: {}".format(e))
            time.sleep(20)
            facebook_book(model)
        except openai.error.APIError as e:
            print("HTTP code 502 from API: {}".format(e))
            time.sleep(20)
            facebook_book(model)
        except Exception as e:
            # Handle other types of exceptions
            print("An error occurred:", e)

        convert_results(checkpoint_dir=f'../data/dataset/facebook_book/google_palm/{m_type}/',
                        output_path=f'../data/dataset/facebook_book/google_palm/output_{m_type}.tsv',
                        m_type=m_type,
                        item_path = '../data/dataset/facebook_book/books.tsv')
    pass


def compute_time(start_time, end_time):
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Format the elapsed time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Display the formatted time
    formatted_time = f"{hours}h {minutes}min {seconds}s"

    return formatted_time


if __name__ == '__main__':
    token = ''
    model = OpenAI('text-davinci-003', token)

    email_host = 'smtp.office365.com'
    email_port = 587
    email_address = ''
    email_password = ''
    receiver_email = ''

    email_subject = 'Python Code Execution Finished'

    message = MIMEMultipart()
    message['From'] = email_address
    message['To'] = receiver_email
    message['Subject'] = email_subject


    server = smtplib.SMTP(email_host, email_port)
    server.starttls()
    server.login(email_address, email_password)

    try:
        # Record the starting time
        start_time = time.time()

        movielens(model)
        hetrec(model)
        facebook_book(model)
        time.sleep(30)

        # Record the ending time
        end_time = time.time()

        email_body = 'Your Python code execution has finished.\nExecuted time: ' + compute_time(start_time, end_time) + '\n\n'
        message.attach(MIMEText(email_body, 'plain'))
        server.send_message(message)
    except Exception as e:
        # Record the ending time
        end_time = time.time()
        email_body = 'Your Python code execution has finished.\nExecuted time: ' + compute_time(start_time, end_time) + '\n\n' + e.args[0] + '\n\n'
        message.attach(MIMEText(email_body, 'plain'))
        server.send_message(message)
        print('Email sent successfully.')
        print(f'Error sending email: {str(e)}')
    finally:
        server.quit()
    pass

import pandas as pd
import typing as t
#from external.elliot.run import run_experiment

"""
    KB Utils
"""
def load_attribute_file(attribute_file, separator='\t'):
    map = {}
    with open(attribute_file) as file:
        for line in file:
            line = line.split(separator)
            int_list = [int(i) for i in line[1:]]
            map[int(line[0])] = list(set(int_list))
    return map
def load_feature_names(infile, separator='\t'):
    feature_names = {}
    with open(infile) as file:
        for line in file:
            line = line.split(separator)
            pattern = line[1].split('><')
            pattern[0] = pattern[0][1:]
            pattern[len(pattern) - 1] = pattern[len(pattern) - 1][:-2]
            feature_names[int(line[0])] = pattern
    return feature_names

"""
    Book and Author Utils
"""
def get_book_name():
    mapping_linked_data = pd.read_csv("../../data/dataset/facebook_book/mappingLinkedData.tsv", sep="\t",
                                      header=None, names=['bookId', 'resourceURI'])
    mapping_linked_data['resourceURI'] = mapping_linked_data['resourceURI'].apply(lambda x: x.split('/')[-1].replace('_', ' '))
    training_set = pd.read_csv("../../data/dataset/facebook_book/trainingset.tsv", sep="\t",
                                      header=None, names=['userId', 'bookId', 'rating'])
    training_set_with_name = pd.merge(training_set, mapping_linked_data, on='bookId').sort_values(by='userId')
    training_set_with_name.to_csv('../../data/dataset/facebook_book/trainingset_with_name.tsv', sep='\t',
                                  header=None, index=False)
    pass
def get_author_name():
    "<http://dbpedia.org/ontology/author>"
    # mapping_linked_data = pd.read_csv("../../data/dataset/facebook_book/mappingLinkedData.tsv", sep="\t",
    #                                   header=None, names=['bookId', 'resourceURI'])
    map = load_attribute_file("../data/dataset/facebook_book/MAPS/map.tsv")

    feature_names = load_feature_names("../data/dataset/facebook_book/features.tsv")

    features_authors = {k:v for k,v in feature_names.items() if "http://dbpedia.org/ontology/author" in v[0]}
    titles_set = set(features_authors.keys())
    mapping_authors_list = {k: [features_authors[el][1].split("/")[-1].replace("_"," ") for el in set(v) & titles_set] for k, v in map.items()}
    # mapping_authors_list['bookId']
    return mapping_authors_list
overall_set = set()
def get_book(genre_list: t.List, threshold: int) -> int:
    acc = 0
    for movie in genre_list:
        if acc >= threshold:
            break
        if movie in overall_set:
            pass
        else:
            acc += 1
            yield movie

"""
    Get Most Popular Book By Genres
"""
def get_most_popular_book():
    # Create a sample dataframe
    df = pd.DataFrame(columns=["bookId", "genreId"])
    book = pd.read_csv('../../data/dataset/facebook_book/trainingset.tsv', sep='\t', header=None,
                          names=['userId', 'bookId', 'rating'])

    with open('../../data/dataset/facebook_book/MAPS/map.tsv', 'r') as file:
        genres = pd.read_csv('../../data/dataset/facebook_book/features_genres.tsv', sep="\t",
                             header=None)[0].to_numpy()
        for line in file:
            pattern = line.strip().split("\t")
            book_id = pattern[0]
            for genre in pattern[1:]:
                if int(genre) in genres:
                    # Append the new row to the dataframe
                    df = df.append({'bookId': book_id, 'genreId': int(genre)}, ignore_index=True)

    threshold = 5
    list_dict = {}

    for genre in df["genreId"].unique():
        genre_books = df[df["genreId"] == genre]["bookId"].tolist()
        ordered_books_by_pop = \
            book[book["bookId"].isin(genre_books)].groupby('bookId')["userId"].count().reset_index("bookId") \
                .sort_values(by="userId", ascending=False)["bookId"].tolist()
        list_dict[genre] = ordered_books_by_pop[:threshold]

    # Select up to 50 unique movies, considering repetitions
    selected_books = set()
    index = 0

    for genre, books in list_dict.items():
        # generator_genre = get_movie(movies, threshold)
        for book in get_book(books, threshold):
            selected_books.add(book)

    # Save the results to a CSV file
    top_books_by_genre = pd.DataFrame(list(selected_books)[:50], columns=['bookId'])
    books = pd.read_csv('../../data/dataset/facebook_book/trainingset_with_name.tsv', sep='\t', header=None,
                       names=['userId', 'bookId', 'rating', 'name'])
    top_books_by_genre = pd.merge(top_books_by_genre, books[['bookId', 'name']], how='left', on='bookId').drop_duplicates()
    top_books_by_genre.to_csv("../../data/dataset/facebook_book/processed_data/top_50_books_by_genre.tsv", sep='\t', index=False)
    pass

def main():
    #run_experiment('../elliot_config_files/movielens/baseline_config_exp_2_re_rank.yml')
    pass

if __name__ == '__main__':
    main()

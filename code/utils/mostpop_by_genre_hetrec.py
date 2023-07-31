import pandas as pd

def add_unique_elements(list1, list2):
    """
    Add elements of list2 not already in list1
    """
    for item in list2:
        if item not in list1:
            list1.append(item)
    return list1

def get_top_artists_by_genre(artists_file, user_artists_file, user_taggedartists_file, tags_file):
    """
    Returns the top 50 most popular artists for the most popular genres based on the user listening history.
    """
    # Load the artists and user_artists files
    artists_df = pd.read_csv(artists_file, sep='\t')
    user_artists_df = pd.read_csv(user_artists_file, sep='\t')
    user_taggedartists_df = pd.read_csv(user_taggedartists_file, sep='\t')
    tags_df = pd.read_csv(tags_file, sep='\t', encoding='latin-1')

    artists_df = pd.merge(artists_df, user_taggedartists_df, how='left', left_on='id', right_on='artistID')

    # Get the count of each genre in the user_taggedartists file
    genre_count_df = pd.merge(user_taggedartists_df, tags_df, on='tagID', how='left')
    # Counting the number of unique user having the tagID
    genre_count_df = genre_count_df.groupby('tagID').agg({'userID': 'nunique'}).reset_index()
    genre_count_df = genre_count_df.rename(columns={'userID': 'count'})
    genre_count_df = genre_count_df.sort_values('count', ascending=False)

    # Get the top genres and corresponding artists
    top_genres = genre_count_df['tagID'].tolist()[:15]  # change to get more or less genres
    top_genre_artists = []
    artist_set = set()
    for genre in top_genres:
        genre_artists_df = pd.merge(user_artists_df, artists_df[['id', 'name', 'tagID']], left_on='artistID', right_on='id')
        # No genre artists have 0
        genre_artists_df['tagID'] = genre_artists_df['tagID'].fillna(0).astype(int)
        # Retrieve the lists of artists by genre
        genre_artists_df = genre_artists_df[genre_artists_df['tagID'] == genre]
        genre_artists_df = genre_artists_df.drop_duplicates(subset=['userID', 'artistID'])
        artist_weights_df = genre_artists_df.groupby('artistID').agg({'weight': 'sum'}).reset_index()
        artist_weights_df = pd.merge(artist_weights_df, artists_df[['id','name']], left_on='artistID', right_on='id').drop_duplicates().drop(columns='id')
        artist_weights_df = artist_weights_df.sort_values('weight', ascending=False).head(15)
        add_unique_elements(top_genre_artists, list(artist_weights_df['artistID']))
    top_genre_artists = pd.DataFrame({'artistID': top_genre_artists})
    top_genre_artists = pd.merge(top_genre_artists['artistID'], artists_df[['id', 'name']], left_on='artistID', right_on='id').drop_duplicates().drop(columns='id')
    #top_genre_artists[:50].to_csv('../../data/dataset/hetrec2011_lastfm_2k/processed_data/top50_most_pop_artists.tsv',
    #                              sep='\t', index=None)

    # Return the top 50 artists and their corresponding weight
    return list(top_genre_artists['name'])[:50]

def create_genre_map(user_taggedartists_file, tags_file, output_file):
    """
    Creates a TSV file containing the mapping between artists and their corresponding genres.
    """
    # Load the tags and user_taggedartists files
    tags_df = pd.read_csv(tags_file, sep='\t', encoding='latin-1')
    user_taggedartists_df = pd.read_csv(user_taggedartists_file, sep='\t')

    # Create a mapping between artists and their corresponding genres
    genre_map_df = pd.merge(user_taggedartists_df, tags_df, on='tagID', how='left')
    genre_map_df = genre_map_df.groupby(['artistID', 'tagValue']).size().unstack(fill_value=0)
    genre_map_df = genre_map_df.reset_index()

    # Write the mapping to the output file
    genre_columns = '\t'.join(genre_map_df.columns[1:])
    genre_map_df.to_csv(output_file, sep='\t', index=False, columns=['artistID'] + list(genre_map_df.columns[1:]))
def create_artist_genre_mapping(tagged_artists_file, output_file):
    tagged_artists_df = pd.read_csv(tagged_artists_file, sep='\t')

    # Group the DataFrame by artistID and aggregate the tagIDs into a list
    artist_genre_map = tagged_artists_df.groupby('artistID')['tagID'].agg(list).reset_index()

    # Write the artist_genre_map DataFrame to a TSV file
    artist_genre_map.to_csv(output_file, sep='\t', index=False, header=False)

if __name__ == "__main__":
    pop_artists_by_genre = get_top_artists_by_genre(artists_file='../../data/dataset/hetrec2011_lastfm_2k/artists.dat',
                              user_artists_file='../../data/dataset/hetrec2011_lastfm_2k/user_artists.dat',
                              user_taggedartists_file='../../data/dataset/hetrec2011_lastfm_2k/user_taggedartists.dat',
                              tags_file='../../data/dataset/hetrec2011_lastfm_2k/tags.dat')

    create_genre_map(user_taggedartists_file='../../data/dataset/hetrec2011_lastfm_2k/user_taggedartists.dat',
                      tags_file='../../data/dataset/hetrec2011_lastfm_2k/tags.dat',
                      output_file='../../data/dataset/hetrec2011_lastfm_2k/map_hetrec.tsv')

    create_artist_genre_mapping(tagged_artists_file='../../data/dataset/hetrec2011_lastfm_2k/user_taggedartists.dat',
                                 output_file='../../data/dataset/hetrec2011_lastfm_2k/map_hetrec.tsv')
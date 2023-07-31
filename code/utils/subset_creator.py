import pandas as pd
import random

# Create a dataframe where the maximum interaction per user are 242
def filter_reviews(df,output_path):
    # Count the number of reviews per user
    user_review_counts = df.groupby('userId').count()['rating']

    # Create a mask for users with more than 230 reviews
    mask = user_review_counts > 200

    # Filter the DataFrame to include only users with more than 242 reviews
    filtered_df = df[df['userId'].isin(user_review_counts[mask].index)]

    # Sort the reviews by rating for each user
    filtered_df = filtered_df.sort_values(['userId', 'rating'], ascending=[True, False])

    # Remove reviews until each user has 242 reviews
    filtered_df = filtered_df.groupby('userId').head(200)

    # Remove filtered users from the initial DataFrame
    initial_df = df[~df['userId'].isin(filtered_df['userId'])]

    # Merge the filtered DataFrame with the initial DataFrame
    merged_df = pd.concat([filtered_df, initial_df]).sort_values('userId')

    # Saving the output into a TSV file
    merged_df.to_csv(output_path, sep='\t', index=False)

    # Return the filtered DataFrame
    return merged_df

# Create and save the sub-dataset where the maximum interaction per user are 242
def filter_the_dataset():
    data = pd.read_csv('../data/dataset/ml_small_2018/splitting/0/train.tsv', sep='\t',
                       names=['userId', 'movieId', 'rating', 'timestamp'], usecols=['userId', 'movieId', 'rating'])

    filter_df = filter_reviews(data, '../data/dataset/ml_small_2018/splitting/0/subset_train_200.tsv')
    print(data, filter_df)
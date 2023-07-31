import pandas as pd

def minmax_scaler_for_users(df):
    # calculate the mean and standard deviation of the column for the specific user
    for user in df['userID'].unique():
        mean = df[df['userID'] == user]['weight'].mean()
        std = df[df['userID'] == user]['weight'].std()
        # normalize the values for the specific user
        df.loc[df['userID'] == user, 'weight'] = \
            (((df.loc[df['userID'] == user, 'weight'] - df.loc[df['userID'] == user, 'weight'].min()) / (df.loc[df['userID'] == user, 'weight'].max() - df.loc[df['userID'] == user, 'weight'].min())) + 0.1) * 5
    return df

def main():
    user_artists = pd.read_csv("../../data/dataset/hetrec2011_lastfm_2k/user_artists.dat", sep="\t")
    normalized_user_artists = minmax_scaler_for_users(user_artists)
    normalized_user_artists.to_csv("../../data/dataset/hetrec2011_lastfm_2k/user_artists_weight.tsv", sep="\t", header=None, index=False)
    pass

if __name__ == '__main__':
    main()
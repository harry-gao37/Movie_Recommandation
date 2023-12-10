import pandas as pd
import requests
import numpy as np

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)



def Recommend_Genre(genre: str):
    data = movies[movies['genres'].str.contains(genre)]

    Genre = data['movie_id'].values.flatten().tolist()  # Flattening the list

    # For each movie, take the average of non-NaN ratings
    df = pd.read_csv('Rmat.csv')
    averaged_df = df.mean(axis=0, skipna=True).reset_index()
    averaged_df['Column_with_prefix'] = averaged_df['index'].apply(lambda x: str(int(x.lstrip('m'))))
    averaged_df.set_index('Column_with_prefix', inplace=True)
    averaged_df.drop("index", axis=1, inplace=True)

    # Filtering out-of-bounds indices
    valid_indices = set(averaged_df.index)
    flattened_list = [str(item) for item in Genre if str(item) in valid_indices]

    # # Then find top 5 comedy movies
    res = averaged_df.loc[flattened_list]
    res.columns = ["value"]
    sorted_indices = res['value'].sort_values(ascending=False).index
    sorted_values = res.loc[sorted_indices].index.tolist()[0:10]
    recommeded_movie = [int(item) for item in sorted_values]
    result = movies[movies['movie_id'].isin(recommeded_movie)]
    return result


def find_pred(row, w):
    # using non na columns to calculate
    common_users = np.where(np.logical_and(row.notna(), w.notna()))[0]
    if len(common_users) == 0:
        return 0
    total = row.iloc[common_users].sum()
    dot_product = np.dot(row.iloc[common_users], w.iloc[common_users])
    return dot_product / total if total != 0 else 0


def myIBCF(newuser):
    S_sorted = pd.read_csv('sorted_data.csv')
    S_sorted.index = S_sorted.columns
    result = S_sorted.apply(find_pred, axis=1, args=(newuser,))
    result = result[result > 0]
    res_sorted = result.sort_values(ascending=False).iloc[0:10]
    res_sorted = res_sorted.reset_index()
    res_sorted = res_sorted['index'].apply(lambda x: str(int(x.lstrip('m'))))
    res_sorted = res_sorted.tolist()
    # if the number of recommended movie less than 10, then we need to find the genre that user has highly rated
    # print(res_sorted)
    if len(res_sorted) < 10:
        highest_rating_movie = newuser.idxmax()
        id = int(highest_rating_movie.lstrip('m'))
        genre = movies[movies['movie_id'] == id]['genres']
        genre = genre.iloc[0].split("|")[0]
        popular_movies = get_popular_movies(str(genre))
        count = 10 - len(res_sorted)
        append_movie = popular_movies[0:count]['movie_id'].tolist()
        append_movie = [str(item) for item in append_movie]
        res_sorted = res_sorted + append_movie
    recommended_movies = res_sorted
    return recommended_movies

def get_displayed_movies():
    return movies.iloc[0:100]

def get_recommended_movies(new_user_input):
    S = pd.read_csv('sorted_data.csv')
    user_input = np.full((S.columns.size), np.nan)
    user_input = pd.Series(user_input, index=S.columns)
    for index, (key, value) in enumerate(new_user_input.items()):
        cur = 'm' + str(key)
        user_input.loc[cur] = value
    recommend_movies = myIBCF(user_input)
    recommend_movies = [int(item) for item in recommend_movies]
    result = movies[movies['movie_id'].isin(recommend_movies)]
    return result

def get_popular_movies(genre: str):
    result = Recommend_Genre(genre)
    return result



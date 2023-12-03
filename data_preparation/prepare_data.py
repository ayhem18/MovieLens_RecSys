"""This script contains the functionalities needed to prepare the data in its initial format for modeling. The process includes the following stages: 
1. preparing the items and users data: renaming the fields, removing extra fields, adding extra fields
2. preparing the ratings data by merging 'items', 'users' and 'ratings', encoding the categorical fields and remove unnecessary columns
"""

import os, sys
import pandas as pd
import itertools

from datetime import datetime
from typing import Union, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# adding the directory 'data_preparation' to the system path so I can refer to it directly
home = os.path.dirname(os.path.realpath(__file__))
current = home
while 'data_preparation' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
DATA_FOLDER = os.path.join(current, 'data')


############################################## PREPARING THE ITEMS CSV ####################################################

movie_genre_index_map = {
0:"unknown",
1:"Action",
2:"Adventure",
3:"Animation",
4:"Children's",
5:"Comedy",
6:"Crime",
7:"Documentary",
8:"Drama",
9:"Fantasy",
10:"Film-Noir",
11:"Horror",
12: "Musical",
13:"Mystery",
14: "Romance",
15: "Sci-Fi",
16: "Thriller",
17: "War",
18: "Western"
}

def _prepare_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function was written to modify the 'date' field in the 'items' dataframe and extract the year as an int
    """
    if 'date' not in list(df.columns):
        raise ValueError(f"Please make sure that 'date' is one of the columns in the passed dataframe.\nFound: {len(df.columns)}")

    df_c = df.copy()
    # first convert the 'date' column from 
    df_c['date'] = pd.to_datetime(df_c['date'], format='mixed', dayfirst=True)
    # a single entry seem to have 'nan' for date. we can simply fill it with the previous year since the films 
    # seem to be grouped by dates in the dataframe
    df_c = df_c.ffill(axis=0)
    
    def extract_year(row) -> pd.Series:
        row['year'] = row['date'].year
        return row

    df_c =  df_c.apply(extract_year, axis='columns').drop(columns=['date'])

    # since all movies are in the range [1900, 2000[, we can use this information to rescale
    # df_c['year'] = (df_c['year'] - 1900) / 100

    return df_c


def prepare_items_csv(items_file_path: Union[str, Path]) -> str:
    # make sure the destination of the file exists
    os.makedirs(os.path.join(DATA_FOLDER, 'prepared'), exist_ok=True)
    
    if os.path.basename(items_file_path) != 'u.item':
        raise ValueError(f"The items data is expected to be saved in a file name 'u.item'. Found: {os.path.basename(items_file_path)}")

    # read the data
    items = pd.read_csv(items_file_path, sep='|', encoding='latin', header=None)

    # drop the title, imdb link, the video release date (all Nan) 
    items.drop(columns=[1, 3, 4], inplace=True)

    # map the initial indices to the names
    index_column_mapper = {0: 'id', 2: 'date'}
    for k, v in movie_genre_index_map.items():
        index_column_mapper[k + 5] = v

    # rename the columns 
    items.rename(columns=index_column_mapper, inplace=True)

    # make sure to modify the 'date' column
    items = _prepare_date_column(items)

    # save the dataframe for later use
    save_path = os.path.join(DATA_FOLDER, 'prepared', "items.csv")
    items.to_csv(save_path, index=False)
    return save_path


############################################## PREPARING THE USERS CSV ####################################################

# the users data is simpler to handle
def prepare_users_csv(users_file_path: Union[str, Path]) -> str:
    os.makedirs(os.path.join(DATA_FOLDER, 'prepared'), exist_ok=True)
    
    if os.path.basename(users_file_path) != 'u.user':
        raise ValueError(f"The users data is expected to be saved in a file name 'u.user'. Found: {os.path.basename(users_file_path)}")

    users = pd.read_csv(users_file_path, sep='|', encoding='latin', header=None)
    users = users.rename(columns={0:'id', 1: 'age', 2: 'gender', 3:'job', 4: 'zip_code'}).drop(columns=['zip_code'])
    # map the gender
    users['gender'] = users['gender'].apply(lambda x: {"M":1, "F":0}[x])
    
    save_path = os.path.join(DATA_FOLDER, 'prepared', "users.csv")

    # make sure to scale the age 
    # scaler = StandardScaler()
    # new_age = scaler.fit_transform(users[['age']].values)
    # users['age'] = new_age.squeeze()
    # users['age'] = users['age'].apply(lambda x: round(x, 6))

    users.to_csv(save_path, index=False)
    return save_path


############################################## PREPARING THE RATINGS CSV ###################################################
def encode_job_genre(df: pd.DataFrame) -> pd.DataFrame:
    # the first step is to remove the 'unknown' key from the genre dictionary as it will skew the results
    del movie_genre_index_map[0]
    genres = list(movie_genre_index_map.values())
    # the next step  
    fields = ['rating', 'job'] + genres
    df_reduced = df[fields]

    # let's build the representation we need
    genre_reps = []

    job_df = pd.pivot_table(df, index='job', aggfunc='count', values='rating')

    for g in genres:
        genre_df = df_reduced[df_reduced[g] == 1]
        # account for the possibility of having no movies with a certain genre rated in the train split
        if len(genre_df) <= 1:
            zeros = [0 for _ in range(len(job_df))]
            genre_reps.append(pd.DataFrame(data={f'count_{g}': zeros, 
                                                 f'mean_{g}': zeros, 
                                                 f'std_{g}': 0}, index=job_df.index))
            continue

        #genre_df: will save the following information: the number of ratings (count), the average of ratings, and the standard deviation
        # of ratings given by each job to a movie with genre: 'g'
        genre_df = pd.pivot_table(genre_df, index=['job'], values='rating', aggfunc=['count', 'mean', 'std'])
        genre_df.columns = [f'count_{g}', f'mean_{g}', f'std_{g}']

        # reduce the total 'count' of ratings given by users with a certain job (to a certain genre) to its percentage by dividing by the
        # total number of ratings given by users with a certain job in general
        genre_df[f'count_{g}'] = genre_df[f'count_{g}'] / job_df['rating']

        # add the genre to the data
        genre_reps.append(genre_df)

    genre_features = pd.concat(genre_reps, axis=1)

    # make sure the resulting dataframe is of the expected shape
    
    exp_shape = (len(df['job'].value_counts()), 3 * len(genres))
    if genre_features.shape != exp_shape:
        raise ValueError(f"The expected shape for the 'job' representations is: {exp_shape}. Found: {genre_features.shape}")

    # now it is time to replace the 'job' column in the original dataframe with the new representation
    # build the dataframe to concatenate horizontally to existing data.
    extra_df =  pd.concat([genre_features.loc[[row['job']], :] for _, row in df.iterrows()], axis=0) 

    if extra_df.shape != (len(df), 3 * len(genres)):
        raise ValueError((f"Please make sure the extra dataframe with new representations of 'job' is of the right shape.\n"
                          f"Expected: {(len(df), 3 * len(genres))}. Found: {extra_df.shape}"))

    extra_df.index = df.index.copy()
    # concatenate the extra dataframe to the original one
    final_df = pd.concat([df.drop(columns=['job']), extra_df], axis=1)
    # make sure to remove the 'job' column at this point
    return final_df


def prepare_model_data(prepared_users_path: Union[str, Path],  
                       prepared_items_path: Union[str, Path], 
                       ratings_path: Union[str, Path],
                       save_path: Union[str, Path],
                       ) -> Tuple[pd.DataFrame, StandardScaler]:
    if os.path.basename(prepared_users_path) != 'users.csv':
        raise ValueError(f"The data is expected to be saved in 'users.csv' file. Found: {os.path.basename(prepare_users_csv)}")

    if os.path.basename(prepared_items_path) != 'items.csv':
        raise ValueError(f"The data is expected to be saved in 'items.csv' file. Found: {os.path.basename(prepared_items_path)}")

    # read the data
    users, items = pd.read_csv(prepared_users_path), pd.read_csv(prepared_items_path)
    
    # read ratings
    ratings = pd.read_csv(ratings_path, sep='\t', encoding='latin', header=None).rename(columns={0: 'user_id', 1: 'item_id', 2: 'rating', 3: 'rating_time'})

    # merge the data
    all_data = pd.merge(left=users, right=ratings, right_on='user_id', left_on='id').drop(columns=['id'])
    all_data = pd.merge(left=all_data, right=items, left_on='item_id', right_on='id').drop(columns=['id'])

    # make sure to sort the dataframe by 'user_id' and then 'rating_time'
    all_data = all_data.sort_values(by=['user_id', 'rating_time']).drop(columns=['rating_time'])

    # time to encode the job in terms of the genres
    all_data = encode_job_genre(all_data)    
    # reorder the columns for easier manipulation 
    final_order = (['user_id', 'item_id', 'age', 'gender'] + 
                   list(itertools.chain(*[[f'count_{g}', f'mean_{g}', f'std_{g}'] for g in list(movie_genre_index_map.values())])) 
                   + ['unknown'] + list(movie_genre_index_map.values()) + ['year', 'rating'])
      
    all_data = all_data[final_order]
    # save the resulting dataframe
    all_data.to_csv(save_path, index=False)
    return all_data


if __name__ == '__main__':
    users, items = os.path.join(DATA_FOLDER, 'ml-100k', 'u.user'), os.path.join(DATA_FOLDER, 'ml-100k', 'u.item'), 
    
    users = prepare_users_csv(users)
    items = prepare_items_csv(items)

    # prepare_model_data(prepared_users_path=users, 
    #                    prepared_items_path=items, 
    #                    ratings_path=os.path.join(DATA_FOLDER, 'ml-100k', 'u1.base'),
    #                    save_path=os.path.join(DATA_FOLDER, 'prepared', f'u1_train.csv')) 

    # prepare_model_data(prepared_users_path=users, 
    #                    prepared_items_path=items, 
    #                    ratings_path=os.path.join(DATA_FOLDER, 'ml-100k', 'u1.test'),
    #                    save_path=os.path.join(DATA_FOLDER, 'prepared', f'u1_test.csv')) 
    

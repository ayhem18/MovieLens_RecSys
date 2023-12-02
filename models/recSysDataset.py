"""
This script contains the functionalities needed to load the data to the model: starting from creating a custom Torch dataset to the dataloaders
"""
import os, sys
import torch
import random

import pandas as pd
import numpy  as np

from torch.utils.data import Dataset, DataLoader
from typing import Union, List, Tuple
from pathlib import Path


home = os.path.dirname(os.path.realpath(__file__))
current = home
while 'models' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
DATA_FOLDER = os.path.join(current, 'data')
print(DATA_FOLDER)

USER_DATA_COLS = ['age', 'gender', 'count_Action', 'mean_Action',
       'std_Action', 'count_Adventure', 'mean_Adventure', 'std_Adventure',
       'count_Animation', 'mean_Animation', 'std_Animation',
       "count_Children's", "mean_Children's", "std_Children's", 'count_Comedy',
       'mean_Comedy', 'std_Comedy', 'count_Crime', 'mean_Crime', 'std_Crime',
       'count_Documentary', 'mean_Documentary', 'std_Documentary',
       'count_Drama', 'mean_Drama', 'std_Drama', 'count_Fantasy',
       'mean_Fantasy', 'std_Fantasy', 'count_Film-Noir', 'mean_Film-Noir',
       'std_Film-Noir', 'count_Horror', 'mean_Horror', 'std_Horror',
       'count_Musical', 'mean_Musical', 'std_Musical', 'count_Mystery',
       'mean_Mystery', 'std_Mystery', 'count_Romance', 'mean_Romance',
       'std_Romance', 'count_Sci-Fi', 'mean_Sci-Fi', 'std_Sci-Fi',
       'count_Thriller', 'mean_Thriller', 'std_Thriller', 'count_War',
       'mean_War', 'std_War', 'count_Western', 'mean_Western', 'std_Western']

ITEM_DATA_COLS = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
       'year']


class RecSysDataset(Dataset):
    def __init__(self, 
                 ratings: Union[pd.DataFrame, str, Path],
                 user_data_cols = List[str], 
                 item_data_cols = List[str],
                 num_pos_samples: int = 5,
                 num_neg_samples: int = 3
                 ):
        self.ratings = ratings if isinstance(ratings, pd.DataFrame) else pd.read_csv(ratings)
        # make sure to extract all items and all users
        self.all_user_ids = set(self.ratings['user_id'])
        self.all_item_ids = set(self.ratings['item_id'])
        self.index2id = {k: v for k, v in enumerate(sorted(list(self.all_user_ids)))}

        # make sure to set the 'user_id' as the index 
        self.ratings.set_index('user_id', inplace=True)
        
        self.num_pos_samples = num_pos_samples
        self.num_neg_samples = num_neg_samples

        self.user_cols = user_data_cols
        self.item_cols = item_data_cols

        expected_layout = ['user_id', 'item_id'] + self.user_cols + self.item_cols + ['rating']
        if list(self.ratings.columns) != expected_layout:
            raise ValueError((f"Please make sure the data layout is as follows : {expected_layout}\n."
                             f"Found: {list(self.ratings.columns)}"))


    def _build_user_history(self, user_ratings: pd.DataFrame, item_ids: List[int]) -> pd.DataFrame:
        # define the user history
        history = np.zeros(shape=(len(item_ids), len(self.item_cols)))
        for index, ii in enumerate(item_ids):
            # make sure the samples have 'item_id' as index
            h = user_ratings.loc[:ii, self.item_cols].values
            ratings = (user_ratings.loc[:ii, 'rating'] / 5).values
            ratings = np.expand_dims(ratings, axis=1) if ratings.ndim == 1 else ratings
            history[index] = np.mean(h * ratings, axis=0)

        return history

    def _positive_sampling(self, user_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        user_ratings = self.ratings.loc[user_id, :]
        # get the items rated by this user
        rated_items = user_ratings['item_id'].tolist()
        pos_items = random.sample(rated_items, self.num_pos_samples)

        samples = user_ratings[user_ratings['item_id'].isin(pos_items)]
        ratings = samples.pop('rating').values
        
        # get the history for the positive samples
        history = self._build_user_history(user_ratings=user_ratings.set_index('item_id'), item_ids=pos_items)
        # concatenate the history to the 
        samples = np.concatenate([np.asarray([[user_id, ii] for ii in pos_items]), 
                                  samples.drop(columns='item_id').values, 
                                  history], 
                                  axis=1)
        return samples, np.ones(shape=(self.num_pos_samples)), ratings

    def _negative_sampling(self, user_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        user_ratings = self.ratings.loc[user_id, :]
        # get the items rated by this user
        rated_items = set(list(user_ratings['item_id']))

        # make sure to get item the user has not see before
        unseen_items = self.all_item_ids.difference(rated_items)
        neg_items = random.sample(list(unseen_items), self.num_neg_samples)

        # the negative samples should be built: 
        # 1. extract the user data from the ratings
        # we do that by extracting the user columns, and then stacking it
        user_data = np.concatenate([user_ratings.loc[:, self.user_cols].iloc[[0], :].values for _ in range(self.num_neg_samples)], axis=0)
        items_data = np.concatenate([self.ratings[self.ratings['item_id'] == ii].loc[:, self.item_cols].iloc[[0], :].values for ii in neg_items], axis=0)

        # get the entire history of the user
        all_history = self._build_user_history(user_ratings=user_ratings.set_index('item_id'), 
                                               item_ids=[user_ratings['item_id'].iloc[-1]])
        
        # expand the history to include all the negative samples
        # concatenate vertically: axis=0
        all_history = np.concatenate([all_history for _ in range(self.num_neg_samples)], axis=0)
        # concatenate horizontally: axis=1
        neg_samples = np.concatenate([np.asarray([[user_id, ii] for ii in neg_items]), 
                                      user_data, 
                                      items_data, 
                                      all_history], 
                                      axis=1)

        return neg_samples, np.zeros(shape=(self.num_neg_samples, )), np.full(shape=(self.num_neg_samples,), fill_value=-1)

    def __len__(self) -> int:
        return len(self.all_user_ids)

    def __getitem__(self, index) -> torch.Tensor:
        # first map the index to the user id`
        user_id = self.index2id[index]
        # concatenate both positive and negative samples                
        pos, pos_c, pos_r = self._positive_sampling(user_id=user_id)
        neg, neg_c, neg_r = self._negative_sampling(user_id=user_id)

        col_nums = 2 + len(self.all_user_ids) + 2 * len(self.all_item_ids)
        # make sure the shapes are as expected
        if neg.shape != (self.num_neg_samples, col_nums) or pos.shape != (self.num_pos_samples, col_nums):
            raise ValueError(f"Expected num columns: {col_nums}. Found: {neg.shape} for negative samples and {pos.shape} for positive samples")

        if pos_c.shape != pos_r.shape or pos_c.shape != (self.num_pos_samples,):
            raise ValueError(f"Expected {(self.num_pos_samples,)} for positive labels. Found: {pos_c.shape}")

        if neg_c.shape != neg_r.shape or neg_c.shape != (self.num_pos_samples,):
            raise ValueError(f"Expected {(self.num_neg_samples,)} for negative labels. Found: {neg.shape}")
    
        # concatenate each of them
        return (torch.from_numpy(np.concatenate([pos, neg])), 
                torch.from_numpy(np.concatenate([pos_c, neg_c], axis=-1)), 
                torch.from_numpy(np.concatenate([pos_r, neg_r], axis=-1))
                )


if __name__ == '__main__':
    dataset = RecSysDataset(ratings=os.path.join(DATA_FOLDER, 'prepared', 'model_data_u1.csv'), 
                        user_data_cols=USER_DATA_COLS,
                        item_data_cols=ITEM_DATA_COLS)  

    train_dataloader = DataLoader(dataset, 
                                  batch_size=10, 
                                  num_workers=2,
                                  shuffle=True,
                                  pin_memory=True)

    next(iter(train_dataloader))     
    

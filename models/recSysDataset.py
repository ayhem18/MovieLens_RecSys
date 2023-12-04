"""
This script contains the functionalities needed to load the data to the model: starting from creating a custom Torch dataset to the dataloaders
"""
import os, sys
import torch
import random

import pandas as pd
import numpy  as np

from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Union, List, Tuple
from pathlib import Path
from tqdm import tqdm


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
                 user_data_cols: List[str], 
                 item_data_cols: List[str],
                 all_users_ids: List[int], 
                 all_items_ids: List[int],
                 num_pos_samples: int = 5,
                 num_neg_samples: int = 3, 
                 negative_sampling: bool = True,
                 use_all_history: bool = False
                 ):
        
        self.ratings = ratings if isinstance(ratings, pd.DataFrame) else pd.read_csv(ratings)
        # make sure to extract all items and all users
        self.all_user_ids = all_users_ids
        self.all_item_ids = all_items_ids
        
        self.user_cols = user_data_cols
        self.item_cols = item_data_cols

        self.data_user_ids = set(self.ratings['user_id'])
        self.data_item_ids = set(self.ratings['item_id'])

        self.user_index2id = {k: v for k, v in enumerate(sorted(list(self.data_user_ids)))}
        self.item_index2id = {k: v for k, v in enumerate(sorted(list(self.data_item_ids)))} 

        self.user_id2index = {v: k for k, v in enumerate(sorted(list(self.data_user_ids)))}
        self.item_id2index = {v: k for k, v in enumerate(sorted(list(self.data_item_ids)))} 


        expected_layout = ['user_id', 'item_id'] + self.user_cols + self.item_cols + ['rating']
        if list(self.ratings.columns) != expected_layout:
            raise ValueError((f"Please make sure the data layout is as follows : {expected_layout}\n."
                             f"Found: {list(self.ratings.columns)}"))


        # make sure to set the 'user_id' as the index 
        self.ratings.set_index('user_id', inplace=True)

        self.num_pos_samples = num_pos_samples
        self.num_neg_samples = num_neg_samples

        self.neg_s = negative_sampling
        self.use_all_history = use_all_history


    def _build_user_history(self, user_ratings: pd.DataFrame, item_ids: List[int]) -> np.ndarray:
        """The function aggregates the the video features for a given users before the given item_id

        Args:
            user_ratings (pd.DataFrame): All the information about the user's rating
            item_ids (List[int]): only items rated before 'item_ids' will be included in the history

        Returns:
            np.ndarray: The representation of the user's history
        """
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
        """Sample positive pairs: (user_id, item_id, user_data, user_history, item_data) 
        where 'user_id' rated 'item_id'

        Args:
            user_id (int): the user for which to sample positive pairs

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: batch of model input, batch of classification labels, batch of regression labels: ratings
        """
        
        user_ratings = self.ratings.loc[[user_id], :]
        # get the items rated by this user
        rated_items = user_ratings['item_id'].tolist()
        
        # there might be cases where there is only one rated item, the 'rated_items' object will be a numpy array
        if isinstance(rated_items, float):
            return np.asarray([[user_id, rated_items]])            

        pos_items = random.sample(rated_items, min(self.num_pos_samples, len(rated_items)))
        samples = user_ratings[user_ratings['item_id'].isin(pos_items)]
        ratings = samples.pop('rating').values
        
        if isinstance(user_ratings, pd.Series):
            user_ratings = user_ratings.to_frame()

        if self.use_all_history:
            all_history = self._build_user_history(user_ratings=user_ratings.set_index('item_id'), 
                                                item_ids=[user_ratings['item_id'].iloc[-1]])
            history = np.concatenate([all_history for _ in pos_items], axis=0)
        else:
            # get the history for the positive samples
            history = self._build_user_history(user_ratings=user_ratings.set_index('item_id'), item_ids=pos_items)
        
        # concatenate the history to the 
        samples = np.concatenate([np.asarray([[user_id, ii] for ii in pos_items]), 
                                  samples.drop(columns='item_id').values, 
                                  history], 
                                  axis=1)
        return np.nan_to_num(samples), np.ones(shape=(samples.shape[0])), ratings

    def _negative_sampling(self, user_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample negative pairs: (user_id, item_id, user_data, user_history, item_data) 
        where 'user_id' did not rate 'item_id' (making sure they are from the given dataset)

        Args:
            user_id (int): the user for which to sample positive pairs

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: batch of model input, batch of classification labels, batch of regression labels: ratings (pseudo labels '-1')
        """

        user_ratings = self.ratings.loc[user_id, :]
        # get the items rated by this user
        rated_items = set(list(user_ratings['item_id']))

        # make sure to get item the user has not see before
        unseen_items = self.data_item_ids.difference(rated_items)
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

        return np.nan_to_num(neg_samples), np.zeros(shape=(neg_samples.shape[0], )), np.full(shape=(neg_samples.shape[0],), fill_value=-1)

    def __len__(self) -> int:
        return len(self.data_user_ids)

    def __getitem__(self, index) -> torch.Tensor:
        """This function maps an index to the user id, then for the given user id carries out
        1. positive sampling
        2. negative sampling
        3. extra checks for data correctness

        Returns:
            torch.Tensor: Concatenation of positive and negative samples and labels
        """
        user_id = self.user_index2id[index]
        # concatenate both positive and negative samples                
        pos, pos_c, pos_r = self._positive_sampling(user_id=user_id)

        if np.isnan(pos).sum() > 0:
            raise ValueError(f"there are nan values in the data. IT will mess up the model !!!")

        if self.neg_s:
            neg, neg_c, neg_r = self._negative_sampling(user_id=user_id)

            col_nums = 2 + len(self.user_cols) + 2 * len(self.item_cols)
            # make sure the shapes are as expected
            if neg.shape[1] != col_nums or pos.shape[1] != col_nums:
                raise ValueError(f"Expected num columns: {col_nums}. Found: {neg.shape} for negative samples and {pos.shape} for positive samples")

            if np.isnan(neg).sum() > 0:
                raise ValueError(f"There are nan values in the negative samples. IT WILL MESS UP THE MODEL !!")

            # concatenate each of them
            return (torch.from_numpy(np.concatenate([pos, neg])), 
                    torch.from_numpy(np.concatenate([pos_c, neg_c], axis=-1)), 
                    torch.from_numpy(np.concatenate([pos_r, neg_r], axis=-1))
                    )

        # at this point return only the positive samples
        return (torch.from_numpy(pos), torch.from_numpy(pos_c), torch.from_numpy(pos_r))


    @classmethod
    def collate_fn(cls, batch):
        """this is a collate function used mainly to convert batch from [3dimensional output to 2 dimensional output] without losing
        the correct order of the data
        Returns:
            The final batch passed to the model
        """
        x, y1, y2 = list(map(list, zip(*batch)))
        return torch.cat(x, dim=0), torch.cat(y1, dim=0), torch.cat(y2, dim=0)


class RecSysInferenceDataset(IterableDataset):
    def __init__(self, 
                 train_ratings: pd.DataFrame, 
                 test_ratings: pd.DataFrame, 
                 all_users_ids: List[int], 
                 all_items_ids: List[int], 
                 user_cols: List[str], 
                 item_cols: List[str], 
                 batch_size: int = 10
                 ) -> None:
        super().__init__()

        self.user_cols = user_cols  
        self.item_cols = item_cols

        self.train_ratings = train_ratings
        self.test_ratings = test_ratings

        expected_layout = ['user_id', 'item_id'] + self.user_cols + self.item_cols + ['rating']
        if list(self.train_ratings.columns) != expected_layout:
            raise ValueError((f"Please make sure the data layout is as follows : {expected_layout}\n."
                             f"Found: {list(self.ratings.columns)}"))

        self.all_user_ids = all_users_ids
        self.all_item_ids = all_items_ids
        
        # build a map between indices and the id
        self.index_2id = {k: v for k, v in enumerate(self.all_user_ids)}
        self.train_ratings.set_index('user_id')

        self.batch_size = batch_size

    def _build_user_history(self, user_ratings: pd.DataFrame, item_ids: List[int]) -> pd.DataFrame:
        """
        same function as the one described above
        """
        # define the user history
        history = np.zeros(shape=(len(item_ids), len(self.item_cols)))
        for index, ii in enumerate(item_ids):
            # make sure the samples have 'item_id' as index
            h = user_ratings.loc[:ii, self.item_cols].values
            ratings = (user_ratings.loc[:ii, 'rating'] / 5).values
            ratings = np.expand_dims(ratings, axis=1) if ratings.ndim == 1 else ratings
            history[index] = np.mean(h * ratings, axis=0)

        return history


    def __iter__(self) -> Tuple[int, torch.Tensor]:
        """
        This function will return an input to the model for all pairs (user_id, item_id) where user_id is in the test model
        """
        # for optimization purposes, use only users in the test set
        test_users = sorted(list(set(self.test_ratings['user_id'])))
        for i in tqdm(test_users):
            # get the data as numpy array
            ui_vector = self.__getitem__(i)
            # divide it into several batches
            for j in range(0, len(ui_vector), self.batch_size):
                yield i, torch.from_numpy(ui_vector[j: j + self.batch_size])


    def __getitem__(self, user_id) -> torch.Tensor:
        """
        This function returs a tensor of the form: (user_id, item_id, user_data, user_history, item_data) for all items in the inference data.
        These tensors are later batched
        """
        
        # extract the all the information about the user ratings
        user_ratings = self.train_ratings.loc[[user_id], :]
        # get all the items the user have already rated
        rated_items = set(user_ratings['item_id'])
        # extract unrated items
        unrated_items = [i_id for i_id in self.all_item_ids if i_id not in rated_items]
        train_items = set(self.train_ratings['item_id'])

        unrated_train_items = [i_id for i_id in train_items if i_id not in rated_items]
        unrated_test_items = [i_id for i_id in self.all_item_ids if (i_id not in train_items)]

        if len(unrated_train_items) + len(unrated_test_items) != len(unrated_items):
            raise ValueError(f"Check your logic. Split the unrated items between train and test") 

        # the user data is ready
        user_data = np.concatenate([user_ratings.loc[:, self.user_cols].iloc[[0], :].values for _ in unrated_items], axis=0)

        train_items_data = np.concatenate([self.train_ratings[self.train_ratings['item_id'] == ii].loc[:, self.item_cols].iloc[[0], :].values for ii in unrated_train_items], axis=0)
        test_items_data = np.concatenate([self.test_ratings[self.test_ratings['item_id'] == ii].loc[:, self.item_cols].iloc[[0], :].values for ii in unrated_test_items], axis=0)

        items_data = np.concatenate([train_items_data, test_items_data], axis=0)

        all_history = self._build_user_history(user_ratings=user_ratings.set_index('item_id'), 
                                               item_ids=[user_ratings['item_id'].iloc[-1]])
        
        # expand the history to include all the unrated items
        # concatenate vertically: axis=0
        all_history = np.concatenate([all_history for _ in unrated_items], axis=0)
        
        items_to_evaluate = np.concatenate([np.asarray([[user_id, ii] for ii in unrated_items]),  
                                user_data, 
                                items_data, 
                                all_history], 
                                axis=1)
        
        return items_to_evaluate



if __name__ == '__main__':

    # an example how to use the dataloaders
    all_user_ids = list(range(1, 944))
    all_item_ids = list(range(1, 1683))

    train_csv = pd.read_csv(os.path.join(DATA_FOLDER, 'prepared', 'u1_train.csv'))
    test_csv = pd.read_csv(os.path.join(DATA_FOLDER, 'prepared', 'u1_test.csv'))

    train_ds = RecSysDataset(ratings=os.path.join(DATA_FOLDER, 'prepared', 'u1_train.csv'), 
                        user_data_cols=USER_DATA_COLS,
                        item_data_cols=ITEM_DATA_COLS, 
                        all_users_ids=all_user_ids,
                        all_items_ids=all_item_ids)  

    train_dataloader = DataLoader(train_ds, 
                                  batch_size=10, 
                                  num_workers=2,
                                  shuffle=True,
                                  pin_memory=True, 
                                  collate_fn=RecSysDataset.collate_fn)

    val_ds = RecSysDataset(ratings=os.path.join(DATA_FOLDER, 'prepared', f'u1_test.csv'),
                            user_data_cols=USER_DATA_COLS, 
                            item_data_cols=ITEM_DATA_COLS,
                            negative_sampling=True,
                            use_all_history=True,
                            all_items_ids=all_item_ids, 
                            all_users_ids=all_user_ids)
    
    val_dl = DataLoader(val_ds,
                        batch_size=10, 
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True,
                        collate_fn=RecSysDataset.collate_fn)
    
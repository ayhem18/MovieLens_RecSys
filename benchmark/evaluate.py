import os
import argparse
import pandas as pd
import torch

from pathlib import Path
from typing import List, Union

DIR = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(Path(DIR), 'data')


def true_user_item(test_ratings: pd.DataFrame,
                   user_id: int, 
                   top_k:int=20) -> List[int]:
    """
    Given the test ratings, the user ids, and the number of items to extract, this function extracts the true labels
    :the items the user liked the most
    
    Args:
        true_items: List of item ids
        recommended_items: list of item 
        top_k (int): The number of items to predict.    
        
    Returns:
        float: The mean average precision.
    """

    # extract all user's ratings
    user_ratings = test_ratings[test_ratings['user_id'] == user_id]
    # consider all those with s
    user_ratings = user_ratings[user_ratings['rating'] >= 4].sort_values(by='rating', ascending=False)

    return user_ratings.iloc[:top_k, :]['item_id'].tolist()


def mean_average_precision(recommended_items: List[int],
                           true_items: List[int], 
                           top_k=20):
    """Computes the mean average precision for a given user.
    
    Args:
        true_items: List of item ids
        recommended_items: list of item 
        top_k (int): The number of items to predict.    
        
    Returns:
        float: The mean average precision.
    """

    # make sure that to extract the top_k elements of both 
    recommended_items = recommended_items[:top_k]
    true_items = true_items[:top_k]

    if len(recommended_items) == 0 or len(true_items) == 0:
        return 0

    num_hits = 0
    total_precision = 0
    for i, item in enumerate(recommended_items):
        if item in true_items:
            num_hits += 1
            total_precision += num_hits / (i + 1)
    
    return total_precision / len(true_items)

def precision_k(true_items: List[int], 
                   recommended_items: List[int], 
                   top_k: int = 20
                   ):
    """Computes the precision at K for a given user.
    
    Args:
        true_items: List of item ids
        recommended_items: list of item 
        top_k (int): The number of items to predict.
        
    Returns:
        float: The precision at K.
    """
    # extract the top_k elements of each list
    true_items = true_items[:top_k]
    recommended_items = recommended_items[:top_k]
    if len(recommended_items) == 0 or len(true_items) == 0:
        return 0
    return len(set(recommended_items) & set(true_items)) / len(recommended_items)


def recall_k(true_items: List[int],
                recommended_items: List[int],
                top_k=25):
    """Computes the recall at K for a given the reommended and true items.
    
    Args:
        true_items: List of item ids
        recommended_items: list of item 
        top_k (int): The number of items to predict.
        
    Returns:
        float: The recall at K.
    """
    
    true_items = true_items[:top_k]
    recommended_items = recommended_items[:top_k]

    if len(recommended_items) == 0 or len(true_items) == 0:
        return 0
    
    return len(set(recommended_items) & set(true_items)) / len(true_items)


def final_evaluation(
        test_ratings: pd.DataFrame,
        recommendatations: pd.DataFrame,
        file_name: Union[str, Path],
        top_k: int = 20):
    # extract the users
    user_ids = sorted(test_ratings['user_id'].tolist())[:-3]

    maps = []
    precisions = []
    recalls = []

    for u_id in user_ids: 
        rec_items = recommendatations.loc[u_id, :].tolist()
        true_items = true_user_item(test_ratings=test_ratings, user_id=u_id, top_k=top_k)

        maps.append(mean_average_precision(rec_items, true_items, top_k=top_k))
        precisions.append(precision_k(rec_items, true_items, top_k=top_k))
        recalls.append(recall_k(rec_items, true_items, top_k=top_k))

        # uncomment for more verbose output
        # print(f'User {u_id}')
        # print(f'MAP: {maps[-1]}')
        # print(f'Precision@K: {precisions[-1]}')
        # print(f'Recall@K: {recalls[-1]}')


    print(f'MAP: {sum(maps) / len(maps) if len(maps) > 0 else 0}')
    print(f'Precision@K: {sum(precisions) / len(precisions) if len(precisions) > 0 else 0}')
    print(f'Recall@K: {sum(recalls) / len(recalls) if len(recalls) > 0 else 0}')

    pd.DataFrame({
        'user_id': user_ids,
        'map': maps,
        'precision': precisions,
        'recall': recalls,
    }).to_csv(os.path.join(Path(DIR).parent, file_name), index=False)

    
if __name__ == '__main__':
    test_ratings = pd.read_csv(os.path.join(DATA_FOLDER, 'u1_test.csv'))
    reg_recs = pd.read_csv(os.path.join(DIR, 'recommendations', 'recommendations_regression.csv'))
    classification_recs = pd.read_csv(os.path.join(DIR, 'recommendations', 'recommendations_classification.csv'))
    
    final_evaluation(test_ratings=test_ratings, recommendatations=reg_recs, file_name='regression_results.csv')
    final_evaluation(test_ratings=test_ratings, recommendatations=classification_recs, file_name='classification_results.csv')

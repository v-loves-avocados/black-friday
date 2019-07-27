#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from googleapiclient import discovery
from google.oauth2 import service_account
from io import BytesIO
from tensorflow.python.lib.io import file_io
import argparse, json, os, uuid, sh
import tensorflow as tf

def initialise_params():
    """Parses all arguments and assigns default values when missing.
    
    Convert argument strings to objects and assign them as attributes of the
    namespace.
    
    Returns:
        An object containing all the parsed arguments for script to use.
    """
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--job_name',
        help='GCS location to write checkpoints and export models.',
        required=True
    )
    args_parser.add_argument(
        '--user_id',
        help='The user ID for prediction.',
        required=True
    )
    args_parser.add_argument(
        '--num_items',
        help='Number of items returned from the prediction.',
        required=True
    )
    args_parser.add_argument(
        '--input_file',
        help='Number of items returned from the prediction.',
        default='gs://reco_exp_1/data/product_df.csv'
    )

    return args_parser.parse_args()


def generate_recommendations(user_idx, user_rated, row_factor, col_factor, k):
    """Generate recommendations for a user.
    
    Args:
    user_idx: the row index of the user in the ratings matrix,

    user_rated: the list of item indexes (column indexes in the ratings matrix)
      previously rated by that user (which will be excluded from the
      recommendations)

    row_factor: the row factors of the recommendation model
    col_factor: the column factors of the recommendation model

    k: number of recommendations requested
    Returns:
    list of k item indexes with the predicted highest rating, excluding
    those that the user has already rated
    """
    
    # bounds checking for args
    #assert (row_factor.shape[0] - len(user_rated)) >= k
    
    # retrieve user factor
    user_f = row_factor[user_idx]
    
    # dot product of item factors with user factor gives predicted ratings
    pred_ratings = col_factor.dot(user_f)
    
    # find candidate recommended item indexes sorted by predicted rating
    k_r = k + len(user_rated)
    candidate_items = np.argsort(pred_ratings)[-k_r:]

    # remove previously rated items and take top k
    recommended_items = [i for i in candidate_items if i not in user_rated]
    recommended_items = recommended_items[-k:]
    
    # flip to sort highest rated first
    recommended_items.reverse()
    
    return recommended_items

def get_recommendations(user_id, num_recs, user_map, item_map, user_factor, item_factor, user_items):
    """Given a user id, return list of num_recs recommended item ids.

    Args:
      user_id: (string) The user id
      num_recs: (int) The number of recommended items to return

    Returns:
      [item_id_0, item_id_1, ... item_id_k-1]: The list of k recommended items,
        if user id is found.
      None: The user id was not found.
    """
    article_recommendations = None

    # map user id into ratings matrix user index
    user_idx = np.searchsorted(user_map, user_id)

    if user_idx is not None:
      # get already viewed items from views dataframe
      already_rated = user_items.get_group(user_id).productId
      already_rated_idx = [np.searchsorted(item_map, i) for i in already_rated]

      # generate list of recommended article indexes from model
      recommendations = generate_recommendations(user_idx, already_rated_idx,
                                                 user_factor,
                                                 item_factor,
                                                 num_recs)

      # map article indexes back to article ids
      article_recommendations = [item_map[i] for i in recommendations]

    return article_recommendations


def main():
    
    parameters = initialise_params()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    # input files

    input_df = pd.read_csv(parameters.input_file, sep=',', header=0)
    df_items = pd.DataFrame({'productId': input_df.productId.unique()})
    df_sorted_items = df_items.sort_values('productId').reset_index()
    pds_items = df_sorted_items.productId
    
    user_items = input_df.groupby('userId')

    profect_id_name = 'hackathon1-183523'
    project_id = 'projects/{}'.format(profect_id_name)
    job_name = 'reco_exp_20190722224244'
    job_id = '{}/jobs/{}'.format(project_id, job_name)

    # Build the service    
    ml = discovery.build('ml', 'v1', cache_discovery=False)

    # Execute the request and pass in the job id
    request = ml.projects().jobs().get(name=job_id).execute()

    # The best model
    best_trial=request['trainingOutput']['trials'][0]['trialId']
    
    job_path='gs://reco_exp_4/job'
    ROW_MODEL_FILE = os.path.join(job_path, best_trial, best_trial, 'model/row.npy')
    COL_MODEL_FILE = os.path.join(job_path, best_trial, best_trial, 'model/col.npy')
    USER_MODEL_FILE = os.path.join(job_path, best_trial, best_trial, 'model/user.npy')
    ITEM_MODEL_FILE = os.path.join(job_path, best_trial, best_trial, 'model/item.npy')
    USER_ITEM_DATA_FILE = 'gs://reco_exp_1/data/product_df.csv'
    
    f = BytesIO(file_io.read_file_to_string(ROW_MODEL_FILE, binary_mode=True))
    user_factor = np.load(f)
    
    f = BytesIO(file_io.read_file_to_string(COL_MODEL_FILE, binary_mode=True))
    item_factor = np.load(f)

    f = BytesIO(file_io.read_file_to_string(USER_MODEL_FILE, binary_mode=True))
    user_map = np.load(f)
    
    f = BytesIO(file_io.read_file_to_string(ITEM_MODEL_FILE, binary_mode=True))
    item_map = np.load(f, allow_pickle=True)

    
    recommended_items = get_recommendations(int(parameters.user_id), int(parameters.num_items), 
                                            user_map, item_map, 
                                            user_factor, item_factor, user_items)
    
    file_name='user_' + str(parameters.user_id) + '.txt'
    out = open(file_name, 'w+')
    for i in recommended_items:
        out.write(i)
        out.write('\n')
    out.close()
    
if __name__ == '__main__':
    main()
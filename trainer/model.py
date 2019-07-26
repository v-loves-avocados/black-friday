# This script follows the example by Google: 
# https://github.com/GoogleCloudPlatform/tensorflow-recommendation-wals

# Format input data, build the model and generate recommendations

import datetime, os, sh, wals
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import tensorflow as tf

# ratio of train set size to test set size
TEST_SET_RATIO = 10

def create_test_and_train_sets(input_file):
    """Load dataset, and create train and set sparse matrices.
    Assumes 'userId', 'productId', and 'Expense' columns.
    Args: 
    input_file: path to the input file
    
    Returns:
    array of user IDs for each row of the ratings matrix
    array of item IDs for each column of the rating matrix
    sparse coo_matrix for training
    sparse coo_matrix for test
    """
    
    input_df = pd.read_csv(input_file, sep=',', header=0)
    df_items = pd.DataFrame({'productId': input_df.productId.unique()})
    df_sorted_items = df_items.sort_values('productId').reset_index()
    pds_items = df_sorted_items.productId
    
    df_user_items = input_df.groupby(['userId', 'productId']).agg({'Expense': 'sum'})
    
    # create a list of (userId, productId, Expense) ratings, where userId and productId are 0-indexed
    current_u = -1
    ux = -1
    pv_ratings = []
    user_ux = []
    
    for timeonpg in df_user_items.itertuples():
        user = timeonpg[0][0]
        item = timeonpg[0][1]
        if user != current_u:
            user_ux.append(user)
            ux += 1
            current_u = user
        ix = pds_items.searchsorted(item)[0]
        pv_ratings.append((ux, ix, timeonpg[1]))

    # convert ratings list and user map to np array
    pv_ratings = np.asarray(pv_ratings)
    user_ux = np.asarray(user_ux)
    
    # create train and test coos matrixes
    tr_sparse, test_sparse = _create_sparse_train_and_test(pv_ratings, ux + 1, df_items.size)
    
    return user_ux, pds_items.as_matrix(), tr_sparse, test_sparse


def _create_sparse_train_and_test(ratings, n_users, n_items):
    """Given ratings, create sparse matrices for train and test sets.
    Args:
    ratings:  list of ratings tuples  (u, i, r)
    n_users:  number of users
    n_items:  number of items
    
    Returns:
    train, test sparse matrices in scipy coo_matrix format.
    """
    
    # pick a random set of data as testing data, sorted ascending
    test_set_size = len(ratings) / TEST_SET_RATIO
    test_set_idx = np.random.choice(xrange(len(ratings)), size=test_set_size, replace=False)
    test_set_idx = sorted(test_set_idx)
    
    # use the remaining data to create a training set
    ts_ratings = ratings[test_set_idx]
    tr_ratings = np.delete(ratings, test_set_idx, axis=0)
    
    # create training and test matrices as coo_matrix
    u_tr, i_tr, r_tr = zip(*tr_ratings)
    tr_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_users, n_items))
    u_ts, i_ts, r_ts = zip(*ts_ratings)
    test_sparse = coo_matrix((r_ts, (u_ts, i_ts)), shape=(n_users, n_items))
    
    return tr_sparse, test_sparse


def train_model(args, tr_sparse):
    """Instantiate WALS model and use "simple_train" to factorize the matrix.
    Args:
    args: a list of parameters
    tr_sparse: sparse training matrix
    
    Returns:
    the row and column factors in numpy format.
    """
    tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    
    # generate model
    input_tensor, row_factor, col_factor, model = wals.wals_model(tr_sparse,
                                                                  args.latent_factors,
                                                                  args.regularization,
                                                                  args.unobs_weight,
                                                                  args.weights,
                                                                  args.wt_type,
                                                                  args.feature_wt_exp,
                                                                  args.feature_wt_factor)
    
    # factorize matrix
    session = wals.simple_train(model, input_tensor, args.num_iters)
    
    tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    
    # evaluate output factor matrices
    output_row = row_factor.eval(session=session)
    output_col = col_factor.eval(session=session)
    
    # close the training session 
    session.close()
    
    return output_row, output_col


def save_model(args, user_map, item_map, row_factor, col_factor):
    
    """Save the user map, item map, row factor and column factor matrices in numpy format.
    
    These matrices together constitute the "recommendation model."
    Args:
    args:         input args to training job
    user_map:     user map numpy array
    item_map:     item map numpy array
    row_factor:   row_factor numpy array
    col_factor:   col_factor numpy array
    """
    
    model_dir = os.path.join(args.output_dir, 'model')
    
    # write model files to /tmp, then copy to GCS
    gs_model_dir = model_dir
    model_dir = '/tmp/{0}'.format(args.job_name)
        
    os.makedirs(model_dir)
    np.save(os.path.join(model_dir, 'user'), user_map)
    np.save(os.path.join(model_dir, 'item'), item_map)
    np.save(os.path.join(model_dir, 'row'), row_factor)
    np.save(os.path.join(model_dir, 'col'), col_factor)
    
    sh.gsutil('cp', '-r', os.path.join(model_dir, '*'), gs_model_dir)

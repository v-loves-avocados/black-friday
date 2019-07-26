# This script follows the example by Google: 
# https://github.com/GoogleCloudPlatform/tensorflow-recommendation-wals
# 

import argparse, json, os
import tensorflow as tf
import model, util, wals

def parse_arguments():
    """Parse job arguments."""
    parser = argparse.ArgumentParser()
    
    # required input arguments
    parser.add_argument(
        '--train-file',
        default='gs://reco_exp_1/data/product_df.csv',
        help='path to training data'
    )
    parser.add_argument(
        '--job-dir',
        required=True,
        help='GCS location to write checkpoints and export models',
    )
    parser.add_argument(
        '--job_name',
        default='rec_exp',
        help='job_name'
    )
    # hyper params for model
    parser.add_argument(
        '--latent_factors',
        type=int,
        default=5,
        help='Number of latent factors'
    )
    parser.add_argument(
        '--num_iters',
        type=int,
        default=20,
        help='Number of iterations for alternating least squares factorization'
    )
    parser.add_argument(
        '--regularization',
        type=float,
        default=0.07,
        help='L2 regularization factor'
    )
    parser.add_argument(
        '--unobs_weight',
        type=float,
        default=0.01,
        help='Weight for unobserved values'
    )
    parser.add_argument(
        '--wt_type',
        type=int,
        default=0,
        help='Rating weight type (0=linear, 1=log)'
    )
    parser.add_argument(
        '--weights',
        type=bool,
        default=True,
        help='If weights should be assigned differently to observe and unobserved items'
    )
    parser.add_argument(
        '--feature_wt_factor',
        type=float,
        default=130,
        help='Feature weight factor (linear ratings)'
    )
    parser.add_argument(
        '--feature_wt_exp',
        type=float,
        default=0.08,
        help='Feature weight exponent (log ratings)'
    )
  
    args = parser.parse_args()
    
    # set output directory for model
    config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    trial = config.get('task', {}).get('trial', '')
    output_dir = os.path.join(args.job_dir, trial)
    
    parser.add_argument(
        '--output_dir',
        default=output_dir,
        help='output directory'
    )    
    
    args_2 = parser.parse_args()
    
    return args_2

def main(args):
    
    tf.logging.set_verbosity(tf.logging.INFO)
 
    # input files
    input_file = util.ensure_local_file(args.train_file)
    user_map, item_map, tr_sparse, test_sparse = model.create_test_and_train_sets(input_file)
    
    # train model
    output_row, output_col = model.train_model(args, tr_sparse)
    
    # save trained model to job directory
    model.save_model(args, user_map, item_map, output_row, output_col)
    
    # log results
    test_rmse = wals.get_rmse(output_row, output_col, test_sparse)
    util.write_hptuning_metric(args, test_rmse)


if __name__ == '__main__':
    job_args = parse_arguments()
    main(job_args)
# This script follows the example by Google: 
# https://github.com/GoogleCloudPlatform/tensorflow-recommendation-wals

import os, uuid, sh
import tensorflow as tf
from tensorflow.core.framework.summary_pb2 import Summary

def ensure_local_file(input_file):
    """
    Ensure the training ratings file is stored locally.
    Args:
    input_file: path to the input file
    
    Returns:
    a local directory
    """
    if input_file.startswith('gs:/'): 
        input_path = os.path.join('/tmp/', str(uuid.uuid4()))
        os.makedirs(input_path)
        tmp_input_file = os.path.join(input_path, os.path.basename(input_file))
        sh.gsutil("cp", "-r", input_file, tmp_input_file)
        return tmp_input_file
    else:
        return input_file


def write_hptuning_metric(args, metric):
    """
    Output a metric measuring the success of the model
    This metric will be used by hypertuning to find the best performing model. 
    Args: 
    args: a list of parameters
    metric: the metric (e.g., test rmse) to be recorded
    
    """
    summary = Summary(value=[Summary.Value(tag='training/hptuning/metric', simple_value=metric)])
    
    # for hyperparam tuning, we write a summary log to a directory 'eval' below the job directory
    eval_path = os.path.join(args.output_dir, 'eval')
    summary_writer = tf.summary.FileWriter(eval_path)
    
    # Note: adding the summary to the writer is enough for hyperparam tuning.
    # The ml engine system is looking for any summary added with the hyperparam metric tag.
    summary_writer.add_summary(summary)
    summary_writer.flush()

from google.cloud import bigquery
import numpy as np
import pandas as pd
import pandas_gbq
import os, uuid, sh
from datetime import datetime
import argparse

client = bigquery.Client()


def parse_arguments():
    """Parse job arguments."""
    parser = argparse.ArgumentParser()
    
    # required input arguments
    parser.add_argument(
        '--project_id',
        default='hackathon1-183523',
        help='project_id'
    )
    parser.add_argument(
        '--dataset_id',
        default='bfdata',
        help='dataset ID'
    )
    parser.add_argument(
        '--table_id',
        default='raw',
        help='Table ID'
    )
    parser.add_argument(
        '--tmp_table_id',
        default='bfdata_tmp' + datetime.now().strftime("%Y%m%d%H%M%S"),
        help='project_id'
    )
    parser.add_argument(
        '--sql',
        default='SELECT User_ID userId, Product_ID productId, Purchase Expense FROM `bfdata.raw`',
        help='SQL for selecting the data'
    )
    parser.add_argument(
        '--gcs_destination',
        default='gs://reco_exp_1/data2/product_df.csv',
        help='GCS storage for the data file'
    )
    
    args = parser.parse_args()
    
    return args


def create_tmp_tb(project_id, dataset_id, tmp_table_id, sql):
    
    '''
    Create a tmp BigQuery table containing the data
    Input: 
        project_id
        dataset_id
        table_id
        tmp_table_id
    Output:
        A temporaty table containing the data
    '''
    # Set the destination table

    job_config = bigquery.QueryJobConfig()
    table_ref = client.dataset(dataset_id).table(tmp_table_id)
    job_config.destination = table_ref
   
    # Start the query, passing in the extra configuration.
    query_job = client.query(
        sql,
        location='US',
        job_config=job_config
    )  

    query_job.result()  # Waits for the query to finish
    print('Query results loaded to table {}'.format(table_ref.path))


def save_gcs(project_id, dataset_id, tmp_table_id, gcs_destination):
    
    '''
    Export BigQuery data to Cloud Storage
    Input:
        project_id
        dataset_id
        table_name
        Destination of the GCS bucket: gs://
    Output: 
        A .csv file saved in the designated bucket
    '''
    
    dataset_ref = client.dataset(dataset_id, project=project_id)
    table_ref = dataset_ref.table(tmp_table_id)

    extract_job = client.extract_table(
        table_ref,
        gcs_destination,
        location="US",
    )  
    extract_job.result()  # Waits for job to complete.

    print(
        "Exported {}:{}.{} to {}".format(project_id, dataset_id, tmp_table_id, gcs_destination)
    )


def main(args):
 
    # create a tmp table in BigQuery
    create_tmp_tb(args.project_id, args.dataset_id, args.tmp_table_id, args.sql)

    # export the tmp table to GCS
    save_gcs(args.project_id, args.dataset_id, args.tmp_table_id, args.gcs_destination)


if __name__ == '__main__':
    job_args = parse_arguments()
    main(job_args)

# Recommendation for Shoppers Using Black Friday Data
# Purpose of the Model
This model uses Kaggle's Black Friday dataset. The original dataset was located at https://www.kaggle.com/mehdidag/black-friday
The modeling method follows Google Cloud's example https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-overview

The purpose of this model is to recommend products for each user ID.

# Data Exploration and Data Analysis
The data analysis and data visualization is done in the Jupyter Notebook bf_exploratory_analysis.ipynb. https://github.com/v-loves-avocados/black-friday/blob/master/bf_exploratory_analysis.ipynb

# Data Preperation
The data_prep.py file handles the data preperation process. 

It does the following three things:
  - Select the columns relevant for the modeling. The default is to include all the data, but this can be customized when executing the file. 
  - Save it in a table in the same BigQuery dataset. The default table name is bfdata_tmp_[timestamp]. The table name can be customized when executing the file. The reason for storing a copy in BigQuery is for SQL users to explore and QA.
  - Move the data from BigQuery to a designated folder in Cloud Storage. There's a default folder, but it could be customized. 
  
~~~
# execute data_prep.py using the default values

python data_prep.py
~~~

~~~
# execute data_prep.py using customized Cloud Storage location

python data_prep.py --gcs_destination='gs://reco_exp_1/data3/product_df.csv'
~~~

The terminal will echo the steps it took.

<img src="/images/terminal.PNG" width="800">

The table with timestamp is created in the Big Query.

<img src="/images/bigquery.PNG" width="400">

The .csv file in Google Cloud Storage:

<img src="/images/gcs.PNG" width="400">



# Modeling
## Input
In the input data, each row is one transaction including the following columns:
  - User ID
  - Product ID
  - Expense
  
## Collaborative Filtering

Collaborative filtering is a popular method to build recommendation systems. It is based on the idea that customers who have similar preferences or tastes in items will respond to them the same way. The gif from https://en.wikipedia.org/wiki/Collaborative_filtering explains the idea.
 
![Alt Text](images/Collaborative_filtering.gif)

## Use Weighted Alternating Least Squares (WALS) to Factorize The Metric

Imagine the transformed input data is transformed into a u (# of users) * i (# of tems) matrix, with each cell listing the corresponding user's preference to the item. 

The goal of the model is to find one such huge matrix that shows the true preference of the users for each product. **The measure of success is the difference between the new matrix that we find, and the input matrix. In this project, we used root mean squared error (rmse) as a metric. **

One "philosophy" in finding such as new matrix is to decompose the input matrix into two parts:

The users' preference towards some latent factors
The items' strength on the latent factors

The latent factors are calculated through a mathematical approach, although it can be given subjective meanings. For example, in our case, we can see that people who purchase a lot of dresses, boots and handbags are heavy on factor 1, we can name the factor 1 as 'fashion'. 

<img src="/images/wals.svg" width="400">
This image is copied from https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-overview. 

Then the model try to fit the training data through an iteration process to find the answer. In the process, there are a few factors that need to be specified:
  - regularization: L2 regularization constant, to avoid overfitting
  - latent_factors: number of latent factors 
  - unobs_weight: weights placed on the unobserved items
  - feature_wt_exp: an exponent number used to scale the weights on the observed items

The WALS method is an algorithm to realize the above process.

## Why WALS?
1. WALS is widely used for implicit ratings. In our case, the $$ spent is an implicit rating.
2. No domain knowledge is needed.
3. It can help discover new insights.
4. Its algorithmic optimization is available.
5. The Tensorflow WALS code base is available, making it quite easy to handle large amount of data.

## Hypertuning
We used Hypertune to tune the following metrics. The goal is to minimize the **Root Mean Squared Error (RMSE)**. The parameters are set in the config.yaml file.
  - regularization: L2 regularization constant, to avoid overfitting
  - latent_factors: number of latent factors 
  - unobs_weight: unobserved item weights
  - feature_wt_exp: feature weight exponent constant

~~~
job=reco_exp_"`date+%Y%m%d%H%M%S`"

gcloud ml-engine jobs submit training $job \
--region=us-central1 \
--module-name=trainer.task \
--package-path=trainer \
--config=config_tune.yaml \
--job-dir='gs://reco_exp_6/job' \
~~~

<img src="/images/job_1.PNG" width="800">
<img src="/images/job_2.PNG" width="800">

After hypertuning, the best performing model came from trial 96, with rmse=2570.058. Since the predicted value is in the unit of cents, this means that the average error for each user is about $25. 


<img src="/images/winning.PNG" width="800">


# Predict

To predict the model, we will run the predictor.py script. The script does three main things:

1. Find the latent factors, the users' and items' weights on the latent factors, from the best performing model based on the results from Hypertune.
2. Discover each user's preference on the items.
3. Output the results in a .txt file in the local directory.

The .py file takes three arguments: 
- job_name: the name of the hypertune job
- user-id: the user of interest
- num_items: the number of items to be returned, preferred items first.

~~~
python predictor.py --job_name xxxxxxxx --user_id xxxxxx –num_items 10
~~~

This outputs a .txt file with the top 10 items that we think the user is most likely to buy.


# Security Concerns
We are extra careful when working with clients' data, especially when PII data is involved. In our project, client id and the products id are anoynymized. Besides, we have the following considerations:

## Secure Infrastructure
We will keep our data, data pipeline and analytics work in the Google Cloud Platform, the most trustworthy and trusted cloud platform.
(https://cloud.google.com/security/infrastructure/)

## Data Loss Prevention
We will work with our client to utilize Google's Cloud Data Loss Prevention service (https://cloud.google.com/dlp/)

## Access Management
We will use Google Cloud's Cloud Identity and Access Management to control users' access to data, and limit the data access only to the personel who need it.  https://cloud.google.com/iam/

## Responsible AI
We will be fully compliant with our company's Respnosible AI requirements, and only use data in the ethical ways. 




References:

1. Google Cloud: https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-overview
2. Kaggle: https://www.kaggle.com/mehdidag/black-friday
3. Wikipedia: https://en.wikipedia.org/wiki/Collaborative_filtering
4. The Security of Google Cloud: https://cloud.google.com/security/

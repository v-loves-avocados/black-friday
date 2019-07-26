# Recommendation for Shoppers Using Black Friday Data
## Purpose of the Model
This model uses Kaggle's Black Friday dataset. The original dataset was located at https://www.kaggle.com/mehdidag/black-friday
The modeling method follows Google Cloud's example https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-overview

The purpose of this model is to recommend products for each user ID.

## Data Exploration and Data Analysis
The data analysis and data visualization is done in the Jupyter Notebook bf_exploratory_analysis.ipynb. https://github.com/v-loves-avocados/black-friday/blob/master/bf_exploratory_analysis.ipynb

# Modeling
## Input
In the input data, each row is one transaction including the following columns:
  - User ID
  - Product ID
  - Expense
  
## Collaborative Filtering

Collaborative filtering is a popular method to build recommendation systems. It predicts the feedback for new items based on large-scale existing user-item feedback. The gif from https://en.wikipedia.org/wiki/Collaborative_filtering explains the idea.
 
![]("images/Collaborative_filtering.gif" width=400)


## Use Weighted Alternating Least Squares (WALS) to Factorize The Metric

Imagine the transformed input data is transformed into a u (# of users) * i (# of tems) matrix, with each cell listing the corresponding user's preference to the item. 

The goal of the model is to find one such huge matrix that shows the true preference of the users for each product. The measure of success is the difference between the new matrix that we find, and the input matrix.

One "philosophy" in finding such as new matrix is to decompose the input matrix into two parts:

The users' preference towards some latent factors
The items' strength on the latent factors

The latent factors are calculated through a mathematical approach, although it can be given subjective meanings. For example, in our case, we can see that people who purchase a lot of dresses, boots and handbags are heavy on factor 1, we can name the factor 1 as 'fashion'. 

<img src="/images/wals.svg" width="400">
This image is copied from https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-overview. 


The WALS method is an algorithm to realize the above process.


~~~
job=reco_exp_"`date+%Y%m%d%H%M%S`"

gcloud ml-engine jobs submit training $job \
--region=us-central1 \
--module-name=trainer.task \
--package-path=trainer \
--config=config_tune.yaml \
--job-dir='gs://reco_exp_4/job' \
-- \
--hypertune
~~~

<img src="/images/job_1.PNG" width="800">
<img src="/images/job_2.PNG" width="800">

~~~
python predictor.py --job_name reco_exp_20190725233820 --user_id 10000001 –num_items 10
~~~

# Recommendation for Shoppers Using Black Friday Data
## Purpose of the Model
This model uses Kaggle's Black Friday dataset. The original dataset was located at https://www.kaggle.com/mehdidag/black-friday

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
 
![](images/Collaborative_filtering.gif)


## Weighted Alternating Least Squares (WALS) Algorithm






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


~~~
python predictor.py --job_name reco_exp_20190725233820 --user_id 10000001 –num_items 10
~~~

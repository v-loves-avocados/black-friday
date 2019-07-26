# black-friday


Submit jobs

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

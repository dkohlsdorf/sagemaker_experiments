'''
sagemaker-user@default:~/sagemaker_experiments/pipelines/demand_prediction_ar_sagemaker$ python training_job.py --train_data s3://sagemaker-eu-north-1-354918397522/sagemaker-scikit-learn-2024-12-16-17-09-19-386/output/train_data/output_pipeline.csv
'''
import argparse
import sagemaker
import boto3

from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput


def main(data):
    sess = sagemaker.Session()
    region = boto3.Session().region_name
    role = sagemaker.get_execution_role()
    processing_instance_type = "ml.m5.xlarge"
    processing_instance_count = 1
    bucket = sess.default_bucket()
    print(f" >>>>> {bucket} {region}")

    s3_input = TrainingInput(s3_data=data)
    estimator = TensorFlow(
        entry_point='training.py',
        instance_count=1,
        instance_type="ml.c5.9xlarge",
        py_version="py37",
        framework_version="2.3.1",
        role=role,
    )
    estimator.fit(inputs={
        'train': s3_input
    })
    training_job_name = estimator.latest_training_job.name
    print("Training Job Name:  {}".format(training_job_name))
    print("https://console.aws.amazon.com/sagemaker/home?region={}#/jobs/{}".format(
        region, training_job_name))
    print("https://console.aws.amazon.com/cloudwatch/home?region={}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={};streamFilter=typeLogStreamPrefix".format(
        region, training_job_name))
    print("https://s3.console.aws.amazon.com/s3/buckets/{}/{}/?region={}&tab=overview".format(
        bucket, training_job_name, region))


def parse_args():
    parser = argparse.ArgumentParser(description="Process")
    parser.add_argument(
        "--train_data",
        type=str,
    )
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    data = args.train_data
    main(data)
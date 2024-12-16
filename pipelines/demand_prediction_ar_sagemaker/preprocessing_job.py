import boto3
import sagemaker
import time
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor


def schedule():
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()
    region = boto3.Session().region_name

    processing_instance_type = "ml.m5.xlarge"
    processing_instance_count = 1
    bucket = sess.default_bucket()
    print(f" >>>>> {bucket} {role}")

    input_uri=f's3://{bucket}/historic.py'
    print(f" >>>>> {input_uri}")

    processor = SKLearnProcessor(
        framework_version="0.20.0",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
    )
    processor.run(
        code="preprocessing.py",
        inputs=[
            ProcessingInput(
                source=input_uri,
                destination="/opt/ml/processing/input/data/",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data", s3_upload_mode="EndOfJob", source="/opt/ml/processing/output/"
            ),
        ],
        arguments=[
            "--input",
            "/opt/ml/processing/input/data/historic.py",
            "--output",
            f"/opt/ml/processing/output/output_pipeline.csv"
        ],
        logs=True,
        wait=False,
    )

    preprocessing_job_description = processor.jobs[-1].describe()
    output_config = preprocessing_job_description["ProcessingOutputConfig"]
    for output in output_config["Outputs"]:
        if output["OutputName"] == "train_data":
            preprocessed_training_data = output["S3Output"]["S3Uri"]
            print(preprocessed_training_data)

    print("================================")
    print(preprocessing_job_description)
    scikit_processing_job_name = processor.jobs[-1].describe()["ProcessingJobName"]
    print("================================")
    print("https://s3.console.aws.amazon.com/s3/buckets/{}/{}/?region={}&tab=overview".format(bucket, scikit_processing_job_name, region))
    print("https://console.aws.amazon.com/sagemaker/home?region={}#/processing-jobs/{}".format(region, scikit_processing_job_name))
    print("https://console.aws.amazon.com/cloudwatch/home?region={}#logStream:group=/aws/sagemaker/ProcessingJobs;prefix={};streamFilter=typeLogStreamPrefix".format(region, scikit_processing_job_name))



if __name__== '__main__':
    schedule()
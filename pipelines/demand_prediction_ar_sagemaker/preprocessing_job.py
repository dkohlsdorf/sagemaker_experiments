import boto3
import sagemaker
import time
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor


def schedule():
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()

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
                output_name="train_data", source="/opt/ml/processing/output"
            ),
        ],
        arguments=[
            "--input",
            "/opt/ml/processing/input/data/historic.py",
            "--output",
            "/opt/ml/processing/output/x_notebook.csv"
        ]
    )
    preprocessing_job_description = sklearn_processor.jobs[-1].describe()
    output_config = preprocessing_job_description["ProcessingOutputConfig"]
    for output in output_config["Outputs"]:
        if output["OutputName"] == "train_data":
            preprocessed_training_data = output["S3Output"]["S3Uri"]


if __name__== '__main__':
    schedule()
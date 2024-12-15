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

    processor = SKLearnProcessor(
        framework_version="0.20.0",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        max_runtime_in_seconds=7200,
    )
    processor.run(
        code="preprocessing.py",
        inputs=[
            ProcessingInput(
                source='s3://dkohlsdorf-experiments/historic.csv',
                destination="/opt/ml/processing/input/data/",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data", source="/opt/ml/processing/train"
            ),
        ],
        arguments=[
            "--input",
            "/opt/ml/processing/input/data/historical.csv",
            "--output"
            "/opt/ml/processing/output/x_notebook.csv"
        ],
    )
    preprocessing_job_description = sklearn_processor.jobs[-1].describe()


if __name__== '__main__':
    schedule()
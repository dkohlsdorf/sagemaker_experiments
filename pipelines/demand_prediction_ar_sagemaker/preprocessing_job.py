import boto3
import sagemaker
import time
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor


def experiment_name():
    timestamp = int(time.time())
    experiment = Experiment.create(
        experiment_name="Daniels-Demand-{}".format(timestamp),
        description="Demand Prediction",
        sagemaker_boto_client=sm,
    )
    experiment_name = experiment.experiment_name
    return experiment_name


def trial_name():
    timestamp = int(time.time())

    trial = Trial.create(
        trial_name="trial-{}".format(timestamp), experiment_name=experiment_name, sagemaker_boto_client=sm
    )

    trial_name = trial.trial_name
    return trial_name


def experiment_config():
    return {
        "ExperimentName": experiment_name(),
        "TrialName": trial_name(),
        "TrialComponentDisplayName": "prepare",
    }


def schedule():
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()

    processing_instance_type = "ml.t3.medium"
    processing_instance_count = 1
    region = boto3.Session().region_name

    print(f"{role} {region}")

    sm = boto3.Session().client(service_name="sagemaker", region_name=region)
    s3 = boto3.Session().client(service_name="s3", region_name=region)

    processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        env={"AWS_DEFAULT_REGION": region},
        max_runtime_in_seconds=7200,
    )
    processor.run(
        code="preprocessing.py",
        inputs=[
            ProcessingInput(
                input_name="raw-input-data",
                source='s3://dkohlsdorf-experiments/historical.csv',
                destination="/opt/ml/processing/input/data/",
                s3_data_distribution_type="ShardedByS3Key",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="output-data", s3_upload_mode="EndOfJob", source="/opt/ml/processing/output/"
            ),
        ],
        arguments=[
            "--input",
            "/opt/ml/processing/input/data/historical.csv",
            "--output"
            "/opt/ml/processing/output/x_notebook.csv"
        ],
        experiment_config=experiment_config,
        logs=True,
        wait=False,
    )


if __name__== '__main__':
    schedule()
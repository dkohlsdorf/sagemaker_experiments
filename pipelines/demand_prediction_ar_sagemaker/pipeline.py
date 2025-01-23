import sagemaker
from pprint import pprint

from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker import session


PROCESSING_INSTANCE = "ml.m5.xlarge"
PROCESSING_INSTANCE_COUNT = 1

TRAINING_INSTANCE_COUNT = 1
TRAINING_INSTANCE = "ml.c5.9xlarge"


def input(raw_input_data_s3_uri):
    return ParameterString(
        name="InputData",
        default_value=raw_input_data_s3_uri,
    )


def preprocessing(input, role):
    return ProcessingStep(
        name='FeatureExtraction',
        code="preprocessing.py",
        processor=SKLearnProcessor(
            framework_version="0.20.0",
            role=role,
            instance_type=PROCESSING_INSTANCE,
            instance_count=PROCESSING_INSTANCE_COUNT,
        ),
        inputs = [
            ProcessingInput(
                source=input,
                destination="/opt/ml/processing/input/data/",
            )
        ],
        outputs = [
            ProcessingOutput(
                output_name="train_data", s3_upload_mode="EndOfJob", source="/opt/ml/processing/output/"
            ),
        ],
        job_arguments=[
            "--input",
            "/opt/ml/processing/input/data/historic.py",
            "--output",
            f"/opt/ml/processing/output/output_pipeline.tfrecord"
        ]
    )


def training(data, role):
    s3_input = TrainingInput(s3_data=data)
    return TrainingStep(
        name="Training",
        estimator=TensorFlow(
            entry_point='training.py',
            instance_count=TRAINING_INSTANCE_COUNT,
            instance_type=TRAINING_INSTANCE,
            py_version="py37",
            framework_version="2.3.1",
            role=role,
            input_mode = "File"
        ),
        inputs={'train': s3_input}
    )
    

def create_pipeline(sess, input, role):
    features = preprocessing(input, role) 
    train = training(features.properties.ProcessingOutputConfig.Outputs['train_data'], role)
    return Pipeline(
        name = "DemandPrediction",
        parameters = [input],
        steps= [features, train],
        sagemaker_session=sess,
    )


def main():
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    role = sagemaker.get_execution_role()
    print(f" >>>> {role}")
    raw_input_data_s3_uri = f's3://{bucket}/historic.py'
    pipeline = create_pipeline(sess, input, role)
    pipeline.create(role_arn=role,description="local pipeline example")
    pipeline_arn = response["PipelineArn"]
    print(pipeline_arn)
    execution = pipeline.start(
        parameters=dict(
            InputData=raw_input_data_s3_uri,
        )
    )
    print(execution.arn)
    execution_run = execution.describe()
    pprint(execution_run)


if __name__ == '__main__':
    main()
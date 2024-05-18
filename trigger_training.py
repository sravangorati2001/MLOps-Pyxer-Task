import boto3
import sagemaker
from sagemaker.estimator import Estimator
import os

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# Get the IAM role
role = 'arn:aws:iam::537646564455:role/awssagemaker'

# S3 bucket for output
bucket = 'output-pyxer'

# Path to your training script and input data
entry_point = 'train.py'
source_dir = '.'
train_instance_type = 'ml.m5.large'
train_instance_count = 1
output_path = f's3://{bucket}/output'

# Create an Estimator
estimator = Estimator(
    entry_point=entry_point,
    source_dir=source_dir,
    role=role,
    instance_count=train_instance_count,
    instance_type=train_instance_type,
    output_path=output_path,
    sagemaker_session=sagemaker_session,
    framework_version='2.3',
    py_version='py37'
)

# Start the training job
estimator.fit({
    'train': f's3://{os.environ["TRAIN_BUCKET"]}/train',
    'test': f's3://{os.environ["TEST_BUCKET"]}/test'
})

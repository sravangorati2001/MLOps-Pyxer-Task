import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator

def main():
    role = 'arn:aws:iam::537646564455:role/awssagemaker'
    region = os.environ['AWS_REGION']
    train_bucket = os.environ['TRAIN_BUCKET']
    test_bucket = os.environ['TEST_BUCKET']
    
    # Define the image URI for the estimator
    image_uri = '763104351884.dkr.ecr.us-west-1.amazonaws.com/tensorflow-training:2.3.0-cpu-py37-ubuntu18.04'

    # Define the entry point script
    entry_point = 'train.py'

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        volume_size=30,
        max_run=3600,
        input_mode='File',
        output_path=f's3://{train_bucket}/output',
        sagemaker_session=sagemaker.Session(),
        entry_point=entry_point,
        source_dir='.'
    )

    train_input = sagemaker.inputs.TrainingInput(
        s3_data=f's3://{train_bucket}/',
        content_type='application/x-image'
    )

    test_input = sagemaker.inputs.TrainingInput(
        s3_data=f's3://{test_bucket}/',
        content_type='application/x-image'
    )

    # Start the training job
    estimator.fit({'train': train_input, 'test': test_input})

if __name__ == '__main__':
    main()

import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator

def main():
    role = 'arn:aws:iam::537646564455:role/awssagemaker'
    region = os.environ['AWS_REGION']
    train_bucket = os.environ['TRAIN_BUCKET']
    test_bucket = os.environ['TEST_BUCKET']

    # Debugging: print current directory and contents
    print("Current Directory: ", os.getcwd())
    print("Directory Contents: ", os.listdir('.'))

    # Verify that train.py exists
    if not os.path.exists('train.py'):
        print("Error: train.py not found!")
        return

    # Define the image URI for the estimator
    image_uri = '763104351884.dkr.ecr.us-west-1.amazonaws.com/tensorflow-training:2.3.0-cpu-py37-ubuntu18.04'

    # Define the entry point script
    entry_point = 'train.py'

    # Debugging: Check for all files in the current directory
    for root, dirs, files in os.walk('.'):
        print(root, dirs, files)

    # Define the source directory
    source_dir = '.'

    # Create a temporary directory to copy necessary files
    import shutil
    temp_source_dir = '/tmp/temp_source_dir'
    if os.path.exists(temp_source_dir):
        shutil.rmtree(temp_source_dir)
    os.makedirs(temp_source_dir)
    
    # Copy only necessary files to the temp directory
    shutil.copy('train.py', temp_source_dir)
    shutil.copy('requirements.txt', temp_source_dir)

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type='ml.m4.xlarge',  # Valid instance type
        volume_size=30,
        max_run=3600,
        input_mode='File',
        output_path=f's3://{train_bucket}/output',
        sagemaker_session=sagemaker.Session(),
        entry_point=entry_point,
        source_dir=temp_source_dir  # Use the temp source directory
    )

    train_input = sagemaker.inputs.TrainingInput(
        s3_data=f's3://{train_bucket}/images',
        content_type='application/x-image'
    )

    test_input = sagemaker.inputs.TrainingInput(
        s3_data=f's3://{test_bucket}/images',
        content_type='application/x-image'
    )

    # Start the training job
    estimator.fit({'train': train_input, 'test': test_input})

if __name__ == '__main__':
    main()

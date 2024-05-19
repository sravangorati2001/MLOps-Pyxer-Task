import os
import pandas as pd
import numpy as np
from PIL import Image
import boto3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def download_data_from_s3(bucket, prefix, local_path):
    s3 = boto3.client('s3')
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        for obj in page['Contents']:
            if obj['Key'].endswith('/'):
                continue
            local_file = os.path.join(local_path, obj['Key'].replace(prefix, ''))
            local_dir = os.path.dirname(local_file)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            s3.download_file(bucket, obj['Key'], local_file)

def load_data(data_dir):
    labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    images = []
    labels = []

    for _, row in labels_df.iterrows():
        image_path = os.path.join(data_dir, 'images', row['image'])
        image = Image.open(image_path)
        image = np.array(image)
        images.append(image)
        labels.append(row['label'])

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

if __name__ == "__main__":
    # Access environment variables
    train_bucket = os.environ.get('TRAIN_BUCKET')
    test_bucket = os.environ.get('TEST_BUCKET')
    train_prefix = 'images/'  # Change this to the correct prefix for images
    test_prefix = 'images/'  # Change this to the correct prefix for images
    local_train_dir = '/opt/ml/input/data/train'
    local_test_dir = '/opt/ml/input/data/test'

    # Download data from S3
    download_data_from_s3(train_bucket, train_prefix, local_train_dir)
    download_data_from_s3(test_bucket, test_prefix, local_test_dir)

    # Load training data
    train_images, train_labels = load_data(local_train_dir)

    # Load test data
    test_images, test_labels = load_data(local_test_dir)

    # Normalize data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Create a simple model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=5)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')

    # Save the model
    model_dir = '/opt/ml/model'
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'model.h5'))

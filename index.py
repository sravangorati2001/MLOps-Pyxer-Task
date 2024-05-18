import os
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from PIL import Image

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Shuffle and take only 10,000 training samples
train_indices = np.random.permutation(len(train_images))[:10000]
train_images = train_images[train_indices]
train_labels = train_labels[train_indices]

# Shuffle and take only 2,000 test samples
test_indices = np.random.permutation(len(test_images))[:2000]
test_images = test_images[test_indices]
test_labels = test_labels[test_indices]

# Function to save images and labels
def save_images_and_labels(images, labels, image_dir, label_file):
    os.makedirs(image_dir, exist_ok=True)
    image_paths = []
    label_data = []

    for idx, (image, label) in enumerate(zip(images, labels)):
        # Define image filename
        image_filename = f'{idx}.png'
        image_path = os.path.join(image_dir, image_filename)
        
        # Save image
        img = Image.fromarray(image)
        img.save(image_path)

        # Collect image path and label
        image_paths.append(image_filename)
        label_data.append(label)

    # Save labels to CSV
    label_df = pd.DataFrame({'image': image_paths, 'label': label_data})
    label_df.to_csv(label_file, index=False)

# Save train and test images and labels
save_images_and_labels(train_images, train_labels, 'train/images', 'train/labels.csv')
save_images_and_labels(test_images, test_labels, 'test/images', 'test/labels.csv')

print("Data preparation completed.")

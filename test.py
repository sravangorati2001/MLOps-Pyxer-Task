import os
import numpy as np
import pandas as pd
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from PIL import Image

# Load test images and labels from the CSV file
labels_df = pd.read_csv('test/labels.csv')
image_filenames = labels_df['image'].tolist()
labels = labels_df['label'].values

# Load and preprocess test images
images = []
for image_filename in image_filenames:
    image_path = os.path.join('test/images', image_filename)
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize pixel values
    images.append(img_array)

# Convert images and labels to numpy arrays
images = np.array(images)
labels = utils.to_categorical(labels)

# Load the trained model
model = load_model('model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(images, labels, verbose=0)

print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
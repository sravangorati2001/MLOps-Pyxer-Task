import os
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, utils
from PIL import Image

# Load image filenames and labels from the CSV file
labels_df = pd.read_csv('train/labels.csv')
image_filenames = labels_df['image'].tolist()
labels = labels_df['label'].values

# Load and preprocess images
images = []
for image_filename in image_filenames:
    image_path = os.path.join('train/images', image_filename)
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize pixel values
    images.append(img_array)

# Convert images and labels to numpy arrays
images = np.array(images)
labels = utils.to_categorical(labels)

# Build the model
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(images, labels, epochs=5, batch_size=64)

# Save the trained model
model.save('model.h5')

print("Model trained and saved successfully.")
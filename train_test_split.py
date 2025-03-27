import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Define source folder and target train/test folders
source_folder = "images"
destination_folder = "images2"
train_folder = "train"
test_folder = "test"

# Create train and test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get class names
classes = set()
for _, cls in enumerate(os.listdir(source_folder)):
    cls = cls.split("_")[:-1]
    cls = ' '.join(cls)
    classes.add(cls)

# Create folders for each class name
for cls in classes:
    os.makedirs(os.path.join(destination_folder, cls), exist_ok=True)
    
# Copy images into the respective folder
os.makedirs(destination_folder, exist_ok=True)
for filename in os.listdir(source_folder):
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, " ".join(filename.split("_")[:-1]), filename)

    if os.path.isfile(source_path):
        shutil.copy(source_path, destination_path)

"""
# Assume that each image filename includes the class name (you can adjust this depending on your setup)
all_images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Here, we'll assume that the class label is part of the filename (e.g., "dog_1.jpg", "cat_2.jpg")
# Extract class labels from filenames. You can adjust this depending on how your dataset is organized.
labels = [image.split('_')[0] for image in all_images]  # For example, it assumes the label is the prefix before the '_'

# Perform stratified split (80% train, 20% test)
train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Move images to the respective train/test folders
for image in train_images:
    shutil.move(os.path.join(source_folder, image), os.path.join(train_folder, image))

for image in test_images:
    shutil.move(os.path.join(source_folder, image), os.path.join(test_folder, image))

print(f"Train and Test split complete. {len(train_images)} images in train folder, {len(test_images)} images in test folder.")"
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# set the path to the dataset (Images MRI)
dataset_path = 'C:/Users/zoama/Downloads/BRAIN_TUMOR_PREDICTION'
# Define the training and testing directories in the dataset_path
train_dir = os.path.join(dataset_path, 'C:/Users/zoama/Downloads/BRAIN_TUMOR_PREDICTION/Training')
test_dir = os.path.join(dataset_path, 'C:/Users/zoama/Downloads/BRAIN_TUMOR_PREDICTION/Testing')
# Define the categories (x4)
categories = ["glioma", "meningioma", "notumor", "pituitary"]
# load and preprocess the training  and testing dataset
def train_distribution():
    train_data = []  # store information about the images in each category (Training)
    # iterates through the categories
    for category in categories:
        folder_path = os.path.join(train_dir, category) # access train folder - access each category
        images = os.listdir(folder_path) # list of image file names in that folder
        count = len(images) # count of images in that category
        train_data.append(pd.DataFrame({"Image": images, "Category": [category] * count, "Count": [count] * count}))  # Panda DataFrame

    train_df = pd.concat(train_data, ignore_index=True)  # concatenate all the DataFrames

    # Visualize the distribution of tumor types in the training dataset - Balanced?
    plt.figure(figsize=(8, 6))
    sns.barplot(data=train_df, x="Category", y="Count")
    plt.title("Distribution of Tumor Types (Testing)")
    plt.xlabel("Tumor Type")
    plt.ylabel("Count")
    plt.show()

    # Visualize sample images for each tumor type
    plt.figure(figsize=(12, 8))
    for i, category in enumerate(categories):
        folder_path = os.path.join(train_dir, category)
        image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
        img = plt.imread(image_path)
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(category)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
def test_distribution():
    test_data = []  # store information about the images in each category (Testing)
    # iterates through the categories
    for category in categories:
        folder_path = os.path.join(test_dir, category) # access train folder - access each category
        images = os.listdir(folder_path) # list of image file names in that folder
        count = len(images) # count of images in that category
        test_data.append(pd.DataFrame({"Image": images, "Category": [category] * count, "Count": [count] * count}))  # Panda DataFrame

    test_df = pd.concat(test_data, ignore_index=True)  # concatenate all the DataFrames

    # Visualize the distribution of tumor types in the training dataset - Balanced?
    plt.figure(figsize=(8, 6))
    sns.barplot(data=test_df, x="Category", y="Count")
    plt.title("Distribution of Tumor Types (Testing)")
    plt.xlabel("Tumor Type")
    plt.ylabel("Count")
    plt.show()

    # Visualize sample images for each tumor type
    plt.figure(figsize=(12, 8))
    for i, category in enumerate(categories):
        folder_path = os.path.join(test_dir, category)
        image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
        img = plt.imread(image_path)
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(category)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

train_distribution() # show the plot and images of the training dataset
test_distribution()  # show the plot and images of the testing dataset

# Set the image size
image_size = (150, 150)  # 150 x 150
# Set the batch size for training - number of samples that will be propagated through the network.
batch_size = 32  # For 5000 samples - 32 batch size is the standard - how is the value of the batch size defined?
# Set the number of epochs for training
epochs = 50

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)


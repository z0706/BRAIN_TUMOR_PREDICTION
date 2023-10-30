from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
#import tensorflow_datasets as tfds - no funciona tensor flow
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the directory containing your training images
train_data_dir = 'C:/Users/zoama/Downloads/BRAIN_TUMOR_PREDICTION/Training'


# Create an ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Rescale pixel values to the [0, 1] range
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift the width of images
    height_shift_range=0.2,  # Randomly shift the height of images
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Randomly zoom in on images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Strategy for filling in newly created pixels
)

# Create the generator for training data
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),  # Resize images to the desired dimensions
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for multi-class problems
)

#model.fit(train_generator, epochs=10, steps_per_epoch=len(train_generator))
'''

# import the modules
from os import listdir

for images in os.listdir(train_data_dir):

    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".png") or images.endswith(".jpg") \
            or images.endswith(".jpeg")):
        # display
        print(images)
'''
#EN SU MAYORIA
# PITUITARY: SAGITAL
# MENINGIOMA: HORIZONTAL
# GLIOMA: HORIZONTAL }
# NO TUMOR: }



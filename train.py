


import numpy as np
import pandas as pd
import os
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, disk
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Load training data
train_df = pd.read_csv('data/train_ship_segmentations.csv')

# Function to decode RLE
def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# Data preparation function
def prepare_data(train_df):
    train_images = []
    train_masks = []

    for idx, row in train_df.iterrows():
        img = imread(f'data/train_images/{row.ImageId}')
        mask = rle_decode(row.EncodedPixels)

        train_images.append(img)
        train_masks.append(mask)

    return np.array(train_images), np.array(train_masks)

X_train, y_train = prepare_data(train_df)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(768, 768, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit model on training data
model.fit(datagen.flow(X_train, y_train, batch_size=16), epochs=10, validation_split=0.1)

# Save the model
model.save('ship_detection_model.h5')

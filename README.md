# Ship-DetectionAI
This project aims to detect and localize ships in satellite images using a trained deep learning model. The model is trained to place aligned bounding boxes around the ships found in the images.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Setup](#setup)
- [Usage](#usage)
- [Training](#training)
- [Prediction](#prediction)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to detect ships in images and place bounding boxes around them. The images can vary in complexity, and ships can be of different sizes and located in various environments such as open sea, docks, and marinas.

## Data

The training data consists of images and their corresponding ground truth annotations provided in run-length encoding (RLE) format. The training images and the RLE annotations are stored in a CSV file (`train_ship_segmentations.csv`).

## Setup

1. **Clone the repository:**

   ```sh
   git clone https://github.com/wagjoker/Ship-DetectionAI.git
   cd Ship-DetectionAI
Install the required dependencies:

Make sure you have  3.7+ and pip installed. Then, run:



pip install -r requirements.txt
Set up the dataset:

Download the dataset and place the images and train_ship_segmentations.csv in a directory called data within the project root.





# Usage
Training
To train the model, you need to prepare the data and then run the training script. Here is an example of how you can do it:

# Prediction
To predict the ship locations in the test images, run the following script:


import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import label, regionprops
from keras.models import load_model

# Load the model
model = load_model('ship_detection_model.h5')

def rle_encode(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

test_images = os.listdir('data/test_images/')
results = []

for img_name in test_images:
    img = imread(f'data/test_images/{img_name}')
    pred_mask = model.predict(img[np.newaxis, ...])[0]
    labeled_mask = label(pred_mask > 0.5)
    
    for region in regionprops(labeled_mask):
        if region.area >= 10:  # Filter small areas
            minr, minc, maxr, maxc = region.bbox
            mask = np.zeros_like(pred_mask, dtype=np.uint8)
            mask[minr:maxr, minc:maxc] = 1
            rle = rle_encode(mask)
            results.append({'ImageId': img_name, 'EncodedPixels': rle})

submission_df = pd.DataFrame(results, columns=['ImageId', 'EncodedPixels'])
submission_df.to_csv('submission.csv', index=False)
##Results
After running the prediction script, the results are saved in submission.csv.

## Contributing
Contributions are welcome! Please fork this repository and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

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

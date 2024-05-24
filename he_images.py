import os
import numpy as np
from PIL import Image
from utils.data_loading import *
import cv2
import argparse
import numpy as np
import glob
from augmentation import *
from hist_eq import *

# Import hist_equalization and ahe functions from hist_eq module
from hist_eq import hist_equalization, ahe

# Define the file paths
file_path = 'data/artifact/images' 
output_path_he = 'data/processed_images/he/images'
output_path_ahe='data/processed_images/ahe/images'
output_path_he_masks = 'data/processed_images/he/masks'
output_path_ahe_masks='data/processed_images/ahe/masks'
# Create the output directory if it doesn't exist
os.makedirs(output_path_he, exist_ok=True)
os.makedirs(output_path_ahe, exist_ok=True)
os.makedirs(output_path_he_masks, exist_ok=True)
os.makedirs(output_path_ahe_masks, exist_ok=True)
# Iterate over the images in the input directory
for image in os.listdir(file_path):
    print(image)
    # Open the image
    img = Image.open(f'{file_path}/{image}')
    # Original file name
    img = np.array(img)
    # Split the original filename into its base name and extension
    base_name, extension = image.rsplit('.', 1)

    # New file name with label
    new_filename = f'{base_name}_label1.{extension}'
    print(new_filename)

    # Open the corresponding mask image
    mask = Image.open(f'data/artifact/masks/{new_filename}')
    mask = np.array(mask)

    # Apply histogram equalization
    hist = hist_equalization(img)
    # Apply adaptive histogram equalization
    ahe_img = ahe(img)

    # Save the processed images
    Image.fromarray(hist).save(f'{output_path_he}/{base_name}_hist.{extension}')
    Image.fromarray(ahe_img).save(f'{output_path_ahe}/{base_name}_ahe.{extension}')
    Image.fromarray(mask).save(f'{output_path_he_masks}/{base_name}_hist_label1.{extension}')
    Image.fromarray(mask).save(f'{output_path_ahe_masks}/{base_name}_ahe_label1.{extension}')
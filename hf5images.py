# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:28:45 2024

@author: vinkjo
"""

import os
import h5py
from PIL import Image
import numpy as np

def convert_images_to_hdf5(image_folder):
    # Ensure the output folder exists
    
    
    # Iterate through each image in the folder
    for root, _, files in os.walk(image_folder):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff','.tif')):
                image_path = os.path.join(root, file_name)
                output_folder = root + '_h5'
                os.makedirs(output_folder, exist_ok=True)
                # Open the image file
                with Image.open(image_path) as img:
                    # Convert the image to an array
                    img_array = np.array(img)
                    
                    # Define the HDF5 filename
                    hdf5_filename = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.h5")
                    
                    # Create an HDF5 file for this image
                    with h5py.File(hdf5_filename, 'w') as hdf5_file:
                        # Create a dataset in the HDF5 file for this image
                        # Using the original filename (with extension) as the dataset name
                        hdf5_file.create_dataset(file_name, data=img_array)

    print("All images have been converted and saved")

# Example usage
image_folder = 'C:/Users/vinkjo/OneDrive - Victoria University of Wellington - STAFF/Desktop/Scott'
convert_images_to_hdf5(image_folder)
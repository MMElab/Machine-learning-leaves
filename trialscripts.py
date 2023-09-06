# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:13:18 2023

@author: vinkjo
"""
import numpy as np
from scipy.spatial import procrustes
def find_transformation_matrix(binary_image1, binary_image2):
    # Ensure both images have the same shape
    assert binary_image1.shape == binary_image2.shape, "Images must have the same shape."

    # Extract coordinates of non-zero (white) pixels from binary images
    coords1 = np.argwhere(binary_image1)
    coords2 = np.argwhere(binary_image2)
    coords2 = coords2[0:len(coords1)]
    # Perform Procrustes analysis
    _, _, transformation = procrustes(coords1, coords2)

    # The transformation matrix is given by the 'rotation' and 'scale' components
    transformation_matrix = transformation["rotation"] * transformation["scale"]

    return transformation_matrix
def detect_orb_keypoints(image):

    # Create the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # Draw the keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints, descriptors, image_with_keypoints

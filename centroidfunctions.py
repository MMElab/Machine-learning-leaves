# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:43:19 2023

@author: vinkjo
"""
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_centroid_leaf(leafimage, leaf):
    a = leafimage.copy()
    a[a!=leaf]=0
    a[a>0]=1
    return calculate_centroid(a)

def calculate_centroid(binary_image):
    # Create an array containing the x-coordinates and y-coordinates of the foreground pixels
    y_coords, x_coords = np.where(binary_image == 1)

    # Calculate the number of foreground pixels
    num_pixels = len(x_coords)

    # Calculate the sum of x-coordinates and y-coordinates
    sum_x = np.sum(x_coords)
    sum_y = np.sum(y_coords)

    # Calculate the centroid (Cx, Cy)
    centroid_x = sum_x / num_pixels
    centroid_y = sum_y / num_pixels

    return centroid_x, centroid_y




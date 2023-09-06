# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:11:52 2023

@author: vinkjo
"""

import numpy as np
from scipy.optimize import minimize

# Generate synthetic point clouds
source_points = np.random.rand(100, 2)  # Replace with your source point cloud data
target_points = source_points.copy()    # Assume a rotated version

# Define the objective function to minimize (distance between points)
def objective_function(angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
    rotated_source_points = source_points @ rotation_matrix.T
    distance = np.sum(np.linalg.norm(rotated_source_points - target_points, axis=1))
    return distance

# Initial guess for the rotation matrix (identity matrix)
initial_rotation_matrix = 0

# Minimize the objective function using optimization
result = minimize(objective_function, initial_rotation_matrix, method='trust-constr', options={'disp': True})

# Extract the optimized rotation matrix
optimized_rotation_matrix = result.x

print("Optimized Rotation Matrix:")
print(optimized_rotation_matrix)
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import h5py
import os
import numpy as np
input_dir = 'C:\\Users\\vinkjo\\Downloads\\OneDrive_2022-03-17\\010322 1 D6'
parent_dir = 'C:\\Users\\vinkjo\\Documents\\h5images'
directory = "010322_1_D6"
output_dir = os.path.join(parent_dir, directory)
#os.mkdir(output_dir)
filenamelist = os.listdir(input_dir)
for filename in filenamelist:
    convert_file(input_dir, filename, output_dir)


def convert_file(input_dir, filename, output_dir):
    filepath = input_dir + '\\' + filename
    fin = open(filepath, 'rb')
    binary_data = fin.read()
    new_filepath = output_dir + '\\' + filename[:-4] + '.hdf5'
    f = h5py.File(new_filepath,'a')
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    dset = f.create_dataset('binary_data', (100, ), dtype=dt)
    dset[0] = np.fromstring(binary_data, dtype='uint8')
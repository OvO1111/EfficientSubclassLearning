import os
import h5py
import json
import numpy as np


path = '/nas/dailinrui/dataset/ACDC'

for _, dirnames, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('.h5'):
           h5 = h5py.File(filename, 'a')
           h5.create_dataset('granularity', (1,), np.uint8, 1, compression='gzip')

mapping = {1: [1, 2, 3]}
with open(os.path.join(path, 'mapping.json'), 'w') as fp:
    json.dump(mapping, fp)
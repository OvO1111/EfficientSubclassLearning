import os
import imageio
import h5py
import json
import numpy as np
import shutil
import SimpleITK as sitk

from functools import wraps
from collections.abc import Iterable
from os.path import *
from tqdm import tqdm

import imgaug as ia

raw_dataset_path = "/nas/dailinrui/dataset/Prostate"
mod_dataset_path = "/data/dailinrui/dataset/prostate"


filename_process = lambda fname: fname.replace('.nii.gz', '.h5')
filelist_process = lambda fname: fname.replace('.nii.gz', '\n')
mask_colors = [0, 255, 510]


def string_modify(str_func):
    def string_mod(func):
        @wraps(func)
        def wrapper_string_mod(*args, **kwargs):
            string_or_string_list = func(*args, **kwargs)
            if isinstance(string_or_string_list, str):
                return str_func(string_or_string_list)
            elif isinstance(string_or_string_list, Iterable) and isinstance(string_or_string_list[0], str):
                return list(str_func(_) for _ in string_or_string_list)
        return wrapper_string_mod
    return string_mod


@string_modify(str_func=filelist_process)
def mod_listdir(dirname):
    return [imname for imname in os.listdir(dirname) if not imname.startswith('.')]


def maybe_mkdir(dirname, filename=None):
    os.makedirs(dirname, exist_ok=True)
    return join(dirname, filename)

all_dirs = mod_listdir(join(raw_dataset_path, 'imagesTr'))
np.random.shuffle(all_dirs)
train_dirs = all_dirs[:24]
test_dirs = all_dirs[24:]
with open(maybe_mkdir(mod_dataset_path, 'train.list'), 'w') as fp:
    fp.writelines(train_dirs)
with open(maybe_mkdir(mod_dataset_path, 'val.list'), 'w') as fp:
    fp.writelines(test_dirs)
mapping = {1: [1, 2]}
with open(maybe_mkdir(mod_dataset_path, 'mapping.json'), 'w') as fp:
    json.dump(mapping, fp)

for imname in tqdm(all_dirs):
    im = join(raw_dataset_path, 'imagesTr', imname.replace('\n', '.nii.gz'))
    mask = im.replace('imagesTr', 'labelsTr')
    if imname.startswith('.'):
        os.remove(im)
        continue
    
    im_ = sitk.GetArrayFromImage(sitk.ReadImage(im))
    mask_ = sitk.GetArrayFromImage(sitk.ReadImage(mask))
    # mask_ = (255 - imageio.imread(mask.replace('.jpg', '.bmp'))) / 127
    
    h5 = h5py.File(maybe_mkdir(join(mod_dataset_path, 'data'), filename_process(imname)), 'w')
    h5.create_dataset('image', im_.shape, np.float32, im_, compression='gzip')
    h5.create_dataset('label', mask_.shape, np.uint8, mask_, compression='gzip')
    h5.create_dataset('granularity', (1,), np.uint8, 1, compression='gzip')

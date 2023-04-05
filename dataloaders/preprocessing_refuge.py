import os
import imageio
import h5py
import numpy as np
import shutil
from functools import wraps
from collections.abc import Iterable
from os.path import *
from tqdm import tqdm

import imgaug as ia

raw_dataset_path = "/nas/dailinrui/dataset/refuge2020"
mod_dataset_path = "/data/dailinrui/dataset/refuge2020"


filename_process = lambda fname: fname.split('_')[0] + '.h5'
filelist_process = lambda fname: fname.split('_')[0] + '\n'
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


with open(maybe_mkdir(mod_dataset_path, 'train.list'), 'w') as fp:
    fp.writelines(mod_listdir(join(raw_dataset_path, 'train', 'images')))
with open(maybe_mkdir(mod_dataset_path, 'val.list'), 'w') as fp:
    fp.writelines(mod_listdir(join(raw_dataset_path, 'valid', 'images')))
with open(maybe_mkdir(mod_dataset_path, 'test.list'), 'w') as fp:
    fp.writelines(mod_listdir(join(raw_dataset_path, 'test', 'images')))

for splits in ['train', 'valid', 'test']:
    for imname in tqdm(os.listdir(join(raw_dataset_path, splits, 'images'))):
        im = join(raw_dataset_path, splits, 'images', imname)
        mask = im.replace('images', 'masks')
        if imname.startswith('.'):
            os.remove(im)
            continue
        
        im_ = imageio.imread(im).transpose(2, 0, 1) / 255
        mask_ = np.sum(imageio.imread(mask), axis=2) / 255
        
        h5 = h5py.File(maybe_mkdir(join(mod_dataset_path, 'data'), filename_process(imname)), 'w')
        h5.create_dataset('image', im_.shape, np.float32, im_, compression='gzip')
        h5.create_dataset('label', mask_.shape, np.uint8, mask_, compression='gzip')

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

coarse_labeling = False
raw_dataset_path = "/nas/dailinrui/dataset/Prostate"
mod_dataset_path = "/nas/dailinrui/dataset/MSDprostate" + ("_coarse" if coarse_labeling else "")
slice_dataset_path = join(mod_dataset_path, 'slices')


filename_process = lambda fname: fname + '.h5'
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
np.random.seed(20001024)
np.random.shuffle(all_dirs)
train_dirs = all_dirs[:24]
train_dirs.sort(key=lambda x:int(x.strip().split('_')[1]))
test_dirs = all_dirs[24:]
with open(maybe_mkdir(mod_dataset_path, 'val.list'), 'w') as fp:
    fp.writelines(test_dirs)
with open(maybe_mkdir(mod_dataset_path, 'test.list'), 'w') as fp:
    fp.writelines(test_dirs)
mapping = {1: [1, 2]}
with open(maybe_mkdir(mod_dataset_path, 'mapping.json'), 'w') as fp:
    json.dump(mapping, fp)

all_dirs = train_dirs.copy()
all_dirs.extend(test_dirs)
fp = open(maybe_mkdir(mod_dataset_path, 'train.list'), 'w')
for iim, imname in tqdm(enumerate(all_dirs)):
    imname = imname.strip()
    im = join(raw_dataset_path, 'imagesTr', imname.replace('\n', '.nii.gz'))
    mask = im.replace('imagesTr', 'labelsTr')
    if imname.startswith('.'):
        os.remove(im)
        continue
    
    im_ = sitk.GetArrayFromImage(sitk.ReadImage(im))
    mask_ = sitk.GetArrayFromImage(sitk.ReadImage(mask))
    print(imname, np.unique(mask_, return_counts=True))
    
    if imname+'\n' in train_dirs:
        if imname+'\n' in train_dirs[:4]:
            print(f'fine:{imname}')
            for s in range(im_.shape[1]):
                fp.write(f"{imname}_slice_{s}\n")
                h5 = h5py.File(maybe_mkdir(join(mod_dataset_path, 'data', 'slices'), filename_process(f"{imname}_slice_{s}")), 'w')
                h5.create_dataset('image', im_[:, s].shape, np.float32, im_[:, s], compression='gzip')
                h5.create_dataset('label', mask_[s].shape, np.uint8, mask_[s], compression='gzip')
                h5.create_dataset('granularity', (1,), np.uint8, 1, compression='gzip')
        else:
            # coarse majorities
            for s in range(im_.shape[1]):
                fp.write(f"{imname}_slice_{s}\n")
                h5 = h5py.File(maybe_mkdir(join(mod_dataset_path, 'data', 'slices'), filename_process(f"{imname}_slice_{s}")), 'w')
                h5.create_dataset('image', im_[:, s].shape, np.float32, im_[:, s], compression='gzip')
                h5.create_dataset('label', mask_[s].shape, np.uint8, (mask_[s] > 0).astype(np.uint8) if coarse_labeling else mask_[s], compression='gzip')
                h5.create_dataset('granularity', (1,), np.uint8, 0 if coarse_labeling else 1, compression='gzip')
    
    else:
        print(f'test:{imname}')
        h5 = h5py.File(maybe_mkdir(join(mod_dataset_path, 'data'), filename_process(imname)), 'w')
        h5.create_dataset('image', im_.shape, np.float32, im_, compression='gzip')
        h5.create_dataset('label', mask_.shape, np.uint8, mask_, compression='gzip')
        h5.create_dataset('granularity', (1,), np.uint8, 1, compression='gzip')
    
fp.close()

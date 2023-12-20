import h5py
import torch
import random
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
from torchvision import transforms
from dataloaders.base_dataset import BaseDataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class Refuge2020(BaseDataset):
    def __init__(self, param, split='train'):
        super(Refuge2020, self).__init__(param, split)
        
        self.normalize = Normalize()
        self.grouped_image_aug_func = BaseImageAffineTransformations()
        self.grouped_tensor_aug_func = BaseImageColorJittering()
        
    def __getitem__(self, idx):
        # sample is randomly cropped and "mixup-ed" in `BaseDataset`
        sample = super().__getitem__(idx)
        
        # return sample

        if self.split == 'train':
            sample = self.grouped_image_aug_func(sample)
            sample = self.grouped_tensor_aug_func(sample)
            
        return sample
    
    
class Normalize(object):
    def __call__(self, sample):
        sample['image'] = sample['image'] / 255
        if sample.__contains__('mixed'):
            sample['mixed'] = sample['mixed'] / 255
        return sample
            
            
class BaseImageAffineTransformations(object):
    def __init__(self, does_flip=0.2, does_rot=0.3):
        self.does_flip = does_flip
        self.does_rot = does_rot
    
    def __call__(self, sample):
        
        does_flip = random.random() > self.does_flip
        does_rot = random.random() > self.does_rot
        flip_axis = random.randint(0, 1)
        rot_angle = random.randint(1, 3)
        
        if does_flip:
            sample['image'] = torch.flip(sample['image'], dims=(flip_axis+1,))
            if sample.__contains__('mixed'):
                sample['mixed'] = torch.flip(sample['mixed'], dims=(flip_axis+1,))
            sample['coarse'] = torch.flip(sample['coarse'], dims=(flip_axis,))
            sample['fine'] = torch.flip(sample['fine'], dims=(flip_axis,))
            
        if does_rot:
            sample['image'] = torch.rot90(sample['image'], dims=(1, 2), k=rot_angle)
            if sample.__contains__('mixed'):
                sample['mixed'] = torch.rot90(sample['mixed'], dims=(1, 2), k=rot_angle)
            sample['coarse'] = torch.rot90(sample['coarse'], dims=(0, 1), k=rot_angle)
            sample['fine'] = torch.rot90(sample['fine'], dims=(0, 1), k=rot_angle)
            
        return sample
            
        
class BaseImageColorJittering(object):
    def __init__(self, does_jitter=0.5):
        self.does_jitter = does_jitter
        
        self.tensor_aug_func = transforms.Compose([
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.2),
                transforms.ColorJitter(contrast=0.2), 
                transforms.ColorJitter(saturation=0.2),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0), 
            ]),
        ])
        
    def __call__(self, sample):
        
        does_jitter = random.random() > self.does_jitter
        if does_jitter:
            sample['image'] = self.tensor_aug_func(sample['image'])
            if sample.__contains__('mixed'):
                sample['mixed'] = self.tensor_aug_func(sample['mixed'])
        
        return sample


def n_choose_2(start, end):
    return random.choices(list(range(start, end+1)), k=2)


AliasDataset = Refuge2020
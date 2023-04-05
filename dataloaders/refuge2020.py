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
    def __init__(self, param, split='train', labeled_idx=None, gray_alpha=1):
        super(Refuge2020, self).__init__(param, split, labeled_idx)
        self.gray_alpha = gray_alpha
        
        self.grouped_image_aug_func = BaseImageAffineTransformations()
        self.grouped_tensor_aug_func = BaseImageColorJittering()
        self.tensor_conversion = ToTensor()
        
    def __getitem__(self, idx):
        # sample is randomly cropped and "mixup-ed" in `BaseDataset`
        sample = super().__getitem__(idx)

        if self.split == 'train':
            sample = self.grouped_image_aug_func(sample)
            sample = self.tensor_conversion(sample)
            sample = self.grouped_tensor_aug_func(sample)
        else:
            sample = self.tensor_conversion(sample)
            
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
            sample['image'] = np.flip(sample['image'], axis=flip_axis+1)
            if sample.__contains__('mixed'):
                sample['mixed'] = np.flip(sample['mixed'], axis=flip_axis+1)
            sample['coarse'] = np.flip(sample['coarse'], axis=flip_axis)
            sample['fine'] = np.flip(sample['fine'], axis=flip_axis)
            
        if does_rot:
            sample['image'] = np.rot90(sample['image'], axes=(1, 2), k=rot_angle)
            if sample.__contains__('mixed'):
                sample['mixed'] = np.rot90(sample['mixed'], axes=(1, 2), k=rot_angle)
            sample['coarse'] = np.rot90(sample['coarse'], axes=(0, 1), k=rot_angle)
            sample['fine'] = np.rot90(sample['fine'], axes=(0, 1), k=rot_angle)
            
        return sample
            
        
class BaseImageColorJittering(object):
    def __init__(self, does_jitter=1):
        self.does_jitter = does_jitter
        
        self.tensor_aug_func = transforms.Compose([
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.2),
                transforms.ColorJitter(contrast=0.2), 
                transforms.ColorJitter(saturation=0.2),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0), 
            ]),
        ])
        self.tensor_conversion = ToTensor()
        
    def __call__(self, sample):
        
        does_jitter = random.random() > self.does_jitter
        if does_jitter:
            sample['image'] = self.tensor_aug_func(sample['image'])
            if sample.__contains__('mixed'):
                sample['mixed'] = self.tensor_aug_func(sample['mixed'])
        
        return sample
            

class ToTensor(object):
    
    def __call__(self, sample):
        sample['image'] = torch.from_numpy(sample['image'].copy()).float()
        if sample.__contains__('mixed'):
            sample['mixed'] = torch.from_numpy(sample['mixed'].copy()).float()
            sample['alpha'] = torch.from_numpy(np.array([sample['alpha'],])).float()
        sample['coarse'] = torch.from_numpy(sample['coarse'].copy()).long()
        sample['fine'] = torch.from_numpy(sample['fine'].copy()).long()
        
        return sample
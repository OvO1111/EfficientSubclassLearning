import h5py
import torch
import random
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
from torchvision import transforms
from dataloaders.base_dataset import BaseDataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class BraTS2021(BaseDataset):
    def __init__(self, param, split='train', labeled_idx=None, gray_alpha=1):
        super(BraTS2021, self).__init__(param, split, labeled_idx)
        self.gray_alpha = gray_alpha
        self.tensor_conversion = ToTensor()
        
    def __getitem__(self, idx):
        # sample is randomly cropped and "mixup-ed" in `BaseDataset`
        sample = super().__getitem__(idx)
        sample = self.tensor_conversion(sample)
            
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
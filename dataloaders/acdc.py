import h5py
import torch
import random
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
from torchvision import transforms
from dataloaders.base_dataset import BaseDataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class ACDC(BaseDataset):
    def __init__(self, param, split='train', labeled_idx=None, gray_alpha=1):
        super(ACDC, self).__init__(param, split, labeled_idx)
        self.gray_alpha = gray_alpha
        
    def __getitem__(self, idx):
        # sample is randomly cropped and "mixup-ed" in `BaseDataset`
        sample = super().__getitem__(idx)
            
        return sample
    
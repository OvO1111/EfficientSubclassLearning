import h5py
import torch
import random
import itertools
import numpy as np
from torch.nn import init
from torch.utils.data import Dataset
from skimage.transform import resize
from torch.utils.data.sampler import Sampler
    
    
class RandomRotFlip(object):

    def __call__(self, sample):
        image, coarse_label, fine_label = sample['image'], sample['lbl:coarse'], sample['lbl:fine']
        ndim = coarse_label.ndim
        assert 1 < ndim < 4
        
        k = np.random.randint(0, 4)
        axis = np.random.randint(1, ndim)
        
        # rot
        image = np.asarray([np.rot90(image[i, ...], k) for i in range(4)])
        coarse_label = np.rot90(coarse_label, k)
        fine_label = np.rot90(fine_label, k)
        
        # flip
        image = np.flip(image, axis=axis).copy()
        coarse_label = np.flip(coarse_label, axis=axis-1).copy()
        fine_label = np.flip(fine_label, axis=axis-1).copy()

        return {'image': image, 'lbl:coarse': coarse_label, 'lbl:fine': fine_label}
    

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, coarse_label, fine_label = sample['image'], sample['lbl:coarse'], sample['lbl:fine']
        ndim = coarse_label.ndim
        assert 1 < ndim < 4
        
        if ndim == 3:
            noise = np.clip(
                self.sigma * np.random.randn(image.shape[1], image.shape[2], image.shape[3]),
                -2*self.sigma,
                2*self.sigma
            )
        elif ndim == 2:
            noise = np.clip(
                self.sigma * np.random.randn(image.shape[1], image.shape[2]),
                -2*self.sigma,
                2*self.sigma
            )
        noise = noise + self.mu
        image = image + noise
        
        return {'image': image, 'lbl:coarse': coarse_label, 'lbl:fine': fine_label}


class TwoStreamBatchSampler(Sampler):

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0, f"{len(self.primary_indices)} >= {self.primary_batch_size} > 0 is not valid"
        assert len(self.secondary_indices) >= self.secondary_batch_size >= 0, f"{len(self.secondary_indices)} >= {self.secondary_batch_size} >= 0 is not valid"

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        if self.secondary_batch_size != 0:
            secondary_iter = iterate_eternally(self.secondary_indices)
            return (
                primary_batch + secondary_batch
                for (primary_batch, secondary_batch)
                in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
            )
        else:
            return (primary_batch for primary_batch in grouper(primary_iter, self.primary_batch_size))

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


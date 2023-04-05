import torch
import numpy as np
import imgaug as ia


class ToTensor(object):

    def __call__(self, sample):
        return {'image': torch.from_numpy(sample['image'].astype(np.float32)),
         'lbl:coarse': torch.from_numpy(sample['lbl:coarse']).long(),
         'lbl:fine': torch.from_numpy(sample['lbl:fine']).long()}



class ImageBaseAugmentation(object):
    pass
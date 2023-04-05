import h5py
import torch
import random
import numpy as np
import imgaug as ia
from torch.utils.data import Dataset
from skimage.transform import resize


class BaseDataset(Dataset):
    
    def __init__(self, param, split='train', labeled_idx=None):
        
        self.split = split
        self.labeled_idx = labeled_idx
        self.mixup = param.exp.mixup_label
        self.num = param.dataset.total_num
        self.patch_size = param.exp.patch_size
        self._base_dir = param.path.path_to_dataset
        self.dataset_name = param.__class__.__name__.replace('Parser', '')

        if self.labeled_idx is None:
            self.labeled_idx = [i for i in range(param.exp.labeled_num)]
        with open(self._base_dir + f'/{split}.list') as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list][:self.num]
        print(f"{self.split}: total {len(self.image_list)} samples")
        
        self.rn_crop = RandomCrop(self.patch_size)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        if self.dataset_name == 'ACDC' and self.split == 'train':
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(image_name), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(image_name), 'r')
        image = h5f['image'][:].copy()
        label_f = h5f['label'][:].copy()
        label_c = h5f['label'][:].copy()
        label_c[label_c > 0] = 1
        
        ndim = label_c.ndim
        assert 1 < ndim < 4
        
        if image.ndim != ndim + 1:
            image = image[np.newaxis, ...]

        sample = {'image': image,
            'coarse': label_c.astype(np.uint8),
            'fine': label_f.astype(np.uint8),
        }
        
        if self.split == 'train' and self.mixup:
            if idx not in self.labeled_idx:
                if ndim == 3: mixed_im, mixed_lf, alpha = self._mixup_ndarray_3d(sample)
                else: mixed_im, mixed_lf, alpha = self._mixup_ndarray_2d(sample)
                
                sample = self.rn_crop(sample)
                mixed_sample = {'image': mixed_im, 'coarse': label_c.astype(np.uint8), 'fine': mixed_lf}
                mixed_sample = self.rn_crop(mixed_sample, keeplast=True)
                sample['fine'] = mixed_sample['fine']
                sample['mixed'] = mixed_sample['image']
                sample['alpha'] = alpha
            else:
                sample = self.rn_crop(sample)
                sample['mixed'] = sample['image']
                sample['alpha'] = 0
        
        elif self.split == 'train':
            sample = self.rn_crop(sample)
            sample['mixed'] = sample['image']
            sample['alpha'] = 0
        else:
            sample = sample
        
        return sample
    
    def _get_bbox_ndarray_2d(self, label_map):
        if np.sum(label_map) == 0:
            return None

        h = np.any(label_map, axis=1)
        w = np.any(label_map, axis=0)

        hmin, hmax = np.where(h)[0][[0, -1]]
        wmin, wmax = np.where(w)[0][[0, -1]]

        return (hmin, hmax+1), (wmin, wmax+1)
    
    def _get_bbox_ndarray_3d(self, label_map):
        if np.sum(label_map) == 0:
            return None

        h = np.any(label_map, axis=(1, 2))
        w = np.any(label_map, axis=(0, 2))
        d = np.any(label_map, axis=(0, 1))

        hmin, hmax = np.where(h)[0][[0, -1]]
        wmin, wmax = np.where(w)[0][[0, -1]]
        dmin, dmax = np.where(d)[0][[0, -1]]

        return (hmin, hmax+1), (wmin, wmax+1), (dmin, dmax+1)
    
    def _mixup_ndarray_2d(self, unlabeled_sample):
        labeled_idx = random.choice(self.labeled_idx)
        q_im, q_lc = unlabeled_sample['image'][:][0], unlabeled_sample['coarse'][:]
        # labeled_h5 = h5py.File(self._base_dir + "/data/{}".format(self.image_list[labeled_idx]), 'r')
        labeled_h5 = h5py.File(self._base_dir + "/data/slices/{}.h5".format(self.image_list[labeled_idx]), "r")
        im = labeled_h5['image'][:]
        lf = labeled_h5['label'][:]
        
        alpha = random.randint(5, 10) / 10
        mixed_im = q_im.copy()
        mixed_lf = np.zeros(q_lc.shape, dtype=np.uint8)
        bbox1 = self._get_bbox_ndarray_2d(lf)
        bbox2 = self._get_bbox_ndarray_2d(q_lc)
        
        if bbox1 is None or bbox2 is None:
            return mixed_im, mixed_lf, 0
        
        cropped_im1 = im[slice(*bbox1[0]), slice(*bbox1[1])]
        cropped_im2 = q_im[slice(*bbox2[0]), slice(*bbox2[1])]
        cropped_lf1 = lf[slice(*bbox1[0]), slice(*bbox1[1])]
        sz_bbox2 = tuple(x[1] - x[0] for x in bbox2)
        rsz_im1 = resize(cropped_im1, output_shape=sz_bbox2, order=1)
        rsz_lf1 = resize(cropped_lf1, output_shape=sz_bbox2, order=0)
        
        rsz_lf1 *= (q_lc[slice(*bbox2[0]), slice(*bbox2[1])] > 0).astype(np.uint8)
        alph = 1 - alpha * (rsz_lf1 > 0).astype(np.uint8)
        mixed_im2 = alph * cropped_im2 + (1 - alph) * rsz_im1
            
        mixed_im[slice(*bbox2[0]), slice(*bbox2[1])] = mixed_im2
        mixed_lf[slice(*bbox2[0]), slice(*bbox2[1])] = rsz_lf1
        
        return np.expand_dims(mixed_im, axis=0), mixed_lf, alpha
    
    def _mixup_ndarray_3d(self, unlabeled_sample):
        labeled_idx = random.choice(self.labeled_idx)
        q_im, q_lc = unlabeled_sample['image'][:], unlabeled_sample['coarse'][:]
        labeled_h5 = h5py.File(self._base_dir + "/data/{}".format(self.image_list[labeled_idx]), 'r')
        im = labeled_h5['image'][:]
        lf = labeled_h5['label'][:]
        
        alpha = random.random()
        mixed_im = q_im.copy()
        mixed_lf = np.zeros(q_lc.shape, dtype=np.uint8)
        bbox1 = self._get_bbox_ndarray_3d(lf)
        bbox2 = self._get_bbox_ndarray_3d(q_lc)
        
        if bbox1 is None or bbox2 is None:
            return mixed_im, mixed_lf, 0
        
        cropped_im1 = im[:, slice(*bbox1[0]), slice(*bbox1[1]), slice(*bbox1[2])]
        cropped_im2 = q_im[:, slice(*bbox2[0]), slice(*bbox2[1]), slice(*bbox2[2])]
        cropped_lf1 = lf[slice(*bbox1[0]), slice(*bbox1[1]), slice(*bbox1[2])]
        sz_bbox2 = tuple(x[1] - x[0] for x in bbox2)
        rsz_im1 = np.concatenate([resize(cropped_im1[channel], output_shape=sz_bbox2, order=3)[np.newaxis, :] for channel in range(4)], axis=0)
        rsz_lf1 = resize(cropped_lf1, output_shape=sz_bbox2, order=0)
        
        rsz_lf1 *= (q_lc[slice(*bbox2[0]), slice(*bbox2[1]), slice(*bbox2[2])] > 0).astype(np.uint8)
        alph = 1 - alpha * (rsz_lf1 > 0).astype(np.uint8)
        mixed_im2 = alph * cropped_im2 + (1 - alph) * rsz_im1
            
        mixed_im[:, slice(*bbox2[0]), slice(*bbox2[1]), slice(*bbox2[2])] = mixed_im2
        mixed_lf[slice(*bbox2[0]), slice(*bbox2[1]), slice(*bbox2[2])] = rsz_lf1
        
        return mixed_im, mixed_lf, alpha


class RandomCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size
        self.w = -1
        self.h = -1
        self.d = -1 

    def __call__(self, sample, keeplast=False):
        image, coarse_label, fine_label = sample['image'], sample['coarse'], sample['fine']
        ndim = coarse_label.ndim
        if image.ndim == ndim:
            image = np.expand_dims(image, axis=0)

        if ndim == 3:
            if coarse_label.shape[0] <= self.output_size[0] or\
                coarse_label.shape[1] <= self.output_size[1] or\
                    coarse_label.shape[2] <= self.output_size[2]:
                pw = max((self.output_size[0] - coarse_label.shape[0]) // 2 + 3, 0)
                ph = max((self.output_size[1] - coarse_label.shape[1]) // 2 + 3, 0)
                pd = max((self.output_size[2] - coarse_label.shape[2]) // 2 + 3, 0)
                image = np.pad(image, [(0, 0), (pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                coarse_label = np.pad(coarse_label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                fine_label = np.pad(fine_label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            (m, w, h, d) = image.shape
            if keeplast:
                w1 = self.w
                h1 = self.h
                d1 = self.d
            else:  
                self.w = w1 = np.random.randint(0, w - self.output_size[0])
                self.h = h1 = np.random.randint(0, h - self.output_size[1])
                self.d = d1 = np.random.randint(0, d - self.output_size[2])

            coarse_label = coarse_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            fine_label = fine_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        
        elif ndim == 2:
            if coarse_label.shape[0] <= self.output_size[0] or\
                coarse_label.shape[1] <= self.output_size[1]:
                pw = max((self.output_size[0] - coarse_label.shape[0]) // 2 + 3, 0)
                ph = max((self.output_size[1] - coarse_label.shape[1]) // 2 + 3, 0)
                image = np.pad(image, [(0, 0), (pw, pw), (ph, ph)], mode='constant', constant_values=0)
                coarse_label = np.pad(coarse_label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
                fine_label = np.pad(fine_label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

            (m, w, h) = image.shape
            if keeplast:
                w1 = self.w
                h1 = self.h
            else:  
                self.w = w1 = np.random.randint(0, w - self.output_size[0])
                self.h = h1 = np.random.randint(0, h - self.output_size[1])

            coarse_label = coarse_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
            fine_label = fine_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
            image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        
        return {'image': image, 'coarse': coarse_label, 'fine': fine_label}
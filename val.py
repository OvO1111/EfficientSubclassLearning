import os
import math
import json
import torch
import shutil
import argparse

import numpy as np
from tqdm import tqdm
from medpy import metric
import SimpleITK as sitk
# from meta import db as param
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict, namedtuple

from networks.singlybranchedunet import UNetSingleBranchNetwork
from utils.visualize import visualize
from dataloaders.base_dataset import BaseDataset

param = None
eval_metrics = [metric.binary.dc, metric.binary.hd95, metric.binary.precision, metric.binary.recall]
n_eval = len(eval_metrics)


def test_single_case(net, parameter, sampled_batch, stride_xy, stride_z=0, gpu_id=None, save_pred=False, ids=None):
    
    global param
    param = parameter
    
    image, gt_c, gt_f =\
        sampled_batch['image'].cuda(gpu_id),\
        sampled_batch['coarse'].detach().numpy(),\
        sampled_batch['fine'].detach().numpy()
    
    if parameter.dataset.n_dim == 3:
        pred_c, pred_f, feat_map = test_single_case_3d(net, image, stride_xy, stride_z, parameter.exp.patch_size)
    elif parameter.dataset.n_dim == 2.5:
        pred_c, pred_f, feat_map = test_single_case_3to2d(net, image, stride_xy, parameter.exp.patch_size)
    elif parameter.dataset.n_dim == 2: 
        pred_c, pred_f, feat_map = test_single_case_2d(net, image, stride_xy, parameter.exp.patch_size)
    
    metric_c, metric_f = np.zeros((parameter.dataset.n_coarse, n_eval)), np.zeros((parameter.dataset.n_fine, n_eval))
    # only evaluate the performance of foreground segmentation
    for c in range(1, parameter.dataset.n_coarse): 
        metric_c[c-1] = calculate_metric_percase(pred_c == c, gt_c == c)
    metric_c[-1] = metric_c[:-1].mean(axis=0)
    for f in range(1, parameter.dataset.n_fine): 
        metric_f[f-1] = calculate_metric_percase(pred_f == f, gt_f == f)
    metric_f[-1] = metric_f[:-1].mean(axis=0)
    
    image = image.squeeze(0)
    
    if save_pred:
        if ids is None:
            ids = 'dummy'
        
        img_save_path = os.path.join(parameter.path.path_to_test, ids)
        os.makedirs(img_save_path, exist_ok=True)
        
        for mode in range(parameter.dataset.n_mode):
            img_itk = sitk.GetImageFromArray(image[mode].cpu().detach().numpy())
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_itk, img_save_path + f"/{ids}_img_mode{mode+1}.nii.gz")

        pred_itk = sitk.GetImageFromArray(pred_f.astype(np.uint8))
        pred_itk.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(pred_itk, img_save_path + f"/{ids}_pred_fine.nii.gz")

        lab_itk = sitk.GetImageFromArray(gt_f.astype(np.uint8))
        lab_itk.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(lab_itk, img_save_path + f"/{ids}_gt_fine.nii.gz")
        
    return metric_c, metric_f, feat_map

    
def test_single_case_2d(net, image, stride_xy, patch_size):
    
    _, _, w, h = image.shape
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    if add_pad:
        image = F.pad(image, (hl_pad, hr_pad, wl_pad, wr_pad), mode='constant')
    
    bb, _, ww, hh = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    score_map_c = np.zeros((bb, param.dataset.n_coarse, ww, hh)).astype(np.float32)
    score_map_f = np.zeros((bb, param.dataset.n_fine, ww, hh)).astype(np.float32)
    feat_map_f = np.zeros((bb, param.network.base_feature_num, ww, hh)).astype(np.float32)
    cnt = np.zeros((bb, ww, hh)).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            test_patch = image[:, :, xs:xs+patch_size[0],ys:ys+patch_size[1]]

            with torch.no_grad():
                out = net(test_patch)
                try:
                    y1c, y1f, feat = out['coarse_logit'], out['fine_logit'], out['feature_map']
                except KeyError:
                    y1f, feat = out['logit'], out['feature_map']
                    y1c = torch.cat([y1f[:, 0:1], y1f[:, 1:].sum(dim=1, keepdim=True)], dim=1)
                yc, yf = torch.softmax(y1c, dim=1), torch.softmax(y1f, dim=1)
                
            yc = yc.cpu().data.numpy()
            yf = yf.cpu().data.numpy()
            feat = feat.cpu().data.numpy()
            
            feat_map_f[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1]] = \
                feat_map_f[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1]] + feat
            
            score_map_c[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                = score_map_c[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1]] + yc
            score_map_f[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                = score_map_f[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1]] + yf
            cnt[:, xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                = cnt[:, xs:xs+patch_size[0], ys:ys+patch_size[1]] + 1
                
    score_map_c = score_map_c/np.expand_dims(cnt, axis=1)
    score_map_f = score_map_f/np.expand_dims(cnt, axis=1)
    feat_map_f = feat_map_f/np.expand_dims(cnt, axis=1)
    label_map_c = np.argmax(score_map_c, axis=1)
    label_map_f = np.argmax(score_map_f, axis=1)
    
    if add_pad:
        label_map_c = label_map_c[:, wl_pad:wl_pad+w, hl_pad:hl_pad+h]
        label_map_f = label_map_f[:, wl_pad:wl_pad+w, hl_pad:hl_pad+h]
        score_map_c = score_map_c[:, :, wl_pad:wl_pad + w, hl_pad:hl_pad+h]
        score_map_f = score_map_f[:, :, wl_pad:wl_pad + w, hl_pad:hl_pad+h]
        feat_map_f = feat_map_f[:, :, wl_pad:wl_pad + w, hl_pad:hl_pad+h]
    
    feat_map_f = np.mean(feat_map_f, axis=0)

    return label_map_c, label_map_f, feat_map_f


def test_single_case_3to2d(net, image, stride_xy, patch_size):
    # special case for 3d images that are sliced to 2d inputs

    _, _, _, h, d = image.shape
    add_pad = False
    if h < patch_size[0]:
        h_pad = patch_size[0]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[1]:
        d_pad = patch_size[1]-d
        add_pad = True
    else:
        d_pad = 0
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = F.pad(image, [dl_pad, dr_pad, hl_pad, hr_pad, 0, 0], mode='constant')
    bb, _, ww, hh, dd = image.shape

    sy = math.ceil((hh - patch_size[0]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[1]) / stride_xy) + 1
    score_map_c = np.zeros((bb, param.dataset.n_coarse, ww, hh, dd)).astype(np.float32)
    score_map_f = np.zeros((bb, param.dataset.n_fine, ww, hh, dd)).astype(np.float32)
    feat_map_f = np.zeros((bb, param.network.base_feature_num, ww, hh, dd)).astype(np.float32)
    cnt = np.zeros((bb, ww, hh, dd)).astype(np.float32)

    for xs in range(0, ww):
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[0])
            for z in range(0, sz):
                zs = min(stride_xy * z, dd-patch_size[1])
                test_patch = image[:, :, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]]

                with torch.no_grad():
                    out = net(test_patch)
                    try:
                        y1c, y1f, feat = out['coarse_logit'], out['fine_logit'], out['feature_map']
                    except KeyError:
                        y1f, feat = out['logit'], out['feature_map']
                        y1c = torch.cat([y1f[:, 0:1], y1f[:, 1:].sum(dim=1, keepdim=True)], dim=1)
                    yc, yf = torch.softmax(y1c, dim=1), torch.softmax(y1f, dim=1)
                    
                yc = yc.cpu().data.numpy()
                yf = yf.cpu().data.numpy()
                feat = feat.cpu().data.numpy()
                
                feat_map_f[:, :, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] = \
                    feat_map_f[:, :, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] + feat
                
                score_map_c[:, :, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] \
                    = score_map_c[:, :, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] + yc
                score_map_f[:, :, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] \
                    = score_map_f[:, :, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] + yf
                cnt[:, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] \
                    = cnt[:, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] + 1
            
            # print((xs, (ys, ys+patch_size[0]), (zs, zs+patch_size[1])), (cnt.sum() / np.prod(cnt.shape),))
                
    score_map_c = score_map_c/np.expand_dims(cnt, axis=1)
    score_map_f = score_map_f/np.expand_dims(cnt, axis=1)
    feat_map_f = feat_map_f/np.expand_dims(cnt, axis=1)
    label_map_c = np.argmax(score_map_c, axis=1)
    label_map_f = np.argmax(score_map_f, axis=1)
    
    if add_pad:
        label_map_c = label_map_c[:, :, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        label_map_f = label_map_f[:, :, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map_c = score_map_c[:, :, :, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map_f = score_map_f[:, :, :, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        feat_map_f = feat_map_f[:, :, :, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    feat_map_f = np.mean(feat_map_f, axis=0)

    return label_map_c, label_map_f, feat_map_f



def test_single_case_3d(net, image, stride_xy, stride_z, patch_size):

    _, _, w, h, d = image.shape
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = F.pad(image, [dl_pad, dr_pad, hl_pad, hr_pad, wl_pad, wr_pad], mode='constant')
    bb, _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map_c = np.zeros((bb, param.dataset.n_coarse, ww, hh, dd)).astype(np.float32)
    score_map_f = np.zeros((bb, param.dataset.n_fine, ww, hh, dd)).astype(np.float32)
    feat_map_f = np.zeros((bb, param.network.base_feature_num, ww, hh, dd)).astype(np.float32)
    cnt = np.zeros((bb, ww, hh, dd)).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[:, :, xs:xs+patch_size[0],ys:ys+patch_size[1], zs:zs+patch_size[2]]

                with torch.no_grad():
                    out = net(test_patch)
                    try:
                        y1c, y1f, feat = out['coarse_logit'], out['fine_logit'], out['feature_map']
                    except KeyError:
                        y1f, feat = out['logit'], out['feature_map']
                        y1c = torch.cat([y1f[:, 0:1], y1f[:, 1:].sum(dim=1, keepdim=True)], dim=1)
                    yc, yf = torch.softmax(y1c, dim=1), torch.softmax(y1f, dim=1)
                    
                yc = yc.cpu().data.numpy()
                yf = yf.cpu().data.numpy()
                feat = feat.cpu().data.numpy()
                
                feat_map_f[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = \
                    feat_map_f[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + feat
                
                score_map_c[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map_c[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + yc
                score_map_f[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map_f[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + yf
                cnt[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
                
    score_map_c = score_map_c/np.expand_dims(cnt, axis=1)
    score_map_f = score_map_f/np.expand_dims(cnt, axis=1)
    feat_map_f = feat_map_f/np.expand_dims(cnt, axis=1)
    label_map_c = np.argmax(score_map_c, axis=1)
    label_map_f = np.argmax(score_map_f, axis=1)
    
    if add_pad:
        label_map_c = label_map_c[:, wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        label_map_f = label_map_f[:, wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map_c = score_map_c[:, :, wl_pad:wl_pad + w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map_f = score_map_f[:, :, wl_pad:wl_pad + w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        feat_map_f = feat_map_f[:, :, wl_pad:wl_pad + w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    feat_map_f = np.mean(feat_map_f, axis=0)

    return label_map_c, label_map_f, feat_map_f


def calculate_metric_percase(pred, gt):
    ret = np.zeros((len(eval_metrics),))
    if pred.sum() > 0 and gt.sum() > 0:
        for i, met in enumerate(eval_metrics):
            ret[i] = met(pred, gt)

    return ret


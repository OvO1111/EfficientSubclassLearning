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

from networks.proposed import CU3D
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
        sampled_batch['coarse'][0].detach().numpy(),\
        sampled_batch['fine'][0].detach().numpy()
    
    if parameter.dataset.n_dim == 3:
        pred_c, pred_f, feat_map = test_single_case_3d(
            net, image, stride_xy, stride_z, parameter.exp.patch_size, (parameter.dataset.n_coarse, parameter.dataset.n_fine)
        )
    elif parameter.dataset.n_dim == 2:
        if 'ACDC' in parameter.__class__.__name__:
            pred_c, pred_f, feat_map = test_single_case_3to2d(
                net, image, stride_xy, parameter.exp.patch_size, (parameter.dataset.n_coarse, parameter.dataset.n_fine)
            )
        else:
            pred_c, pred_f, feat_map = test_single_case_2d(
                net, image, stride_xy, parameter.exp.patch_size, (parameter.dataset.n_coarse, parameter.dataset.n_fine)
            )
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

    
def test_single_case_2d(net, image, stride_xy, patch_size, n_labels):
    
    n_coarse, n_fine = n_labels
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
    
    _, _, ww, hh = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    score_map_c = np.zeros((n_coarse, ww, hh)).astype(np.float32)
    score_map_f = np.zeros((n_fine, ww, hh)).astype(np.float32)
    feat_map_f = np.zeros((param.network.base_feature_num, ww, hh)).astype(np.float32)
    cnt = np.zeros((ww, hh)).astype(np.float32)

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
                
            yc = yc.cpu().data.numpy()[0]
            yf = yf.cpu().data.numpy()[0]
            feat = feat.cpu().data.numpy()[0]
            
            feat_map_f[:, xs:xs+patch_size[0], ys:ys+patch_size[1]] = \
                feat_map_f[:, xs:xs+patch_size[0], ys:ys+patch_size[1]] + feat
            
            score_map_c[:, xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                = score_map_c[:, xs:xs+patch_size[0], ys:ys+patch_size[1]] + yc
            score_map_f[:, xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                = score_map_f[:, xs:xs+patch_size[0], ys:ys+patch_size[1]] + yf
            cnt[xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1]] + 1
                
    score_map_c = score_map_c/np.expand_dims(cnt, axis=0)
    score_map_f = score_map_f/np.expand_dims(cnt, axis=0)
    feat_map_f = feat_map_f/np.expand_dims(cnt, axis=0)
    label_map_c = np.argmax(score_map_c, axis=0)
    label_map_f = np.argmax(score_map_f, axis=0)
    
    if add_pad:
        label_map_c = label_map_c[wl_pad:wl_pad+w, hl_pad:hl_pad+h]
        label_map_f = label_map_f[wl_pad:wl_pad+w, hl_pad:hl_pad+h]
        score_map_c = score_map_c[:, wl_pad:wl_pad + w, hl_pad:hl_pad+h]
        score_map_f = score_map_f[:, wl_pad:wl_pad + w, hl_pad:hl_pad+h]
        feat_map_f = feat_map_f[:, wl_pad:wl_pad + w, hl_pad:hl_pad+h]

    return label_map_c, label_map_f, feat_map_f


def test_single_case_3to2d(net, image, stride_xy, patch_size, n_labels):
    # special case for ACDC
    n_coarse, n_fine = n_labels
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
    _, _, ww, hh, dd = image.shape

    sy = math.ceil((hh - patch_size[0]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[1]) / stride_xy) + 1
    score_map_c = np.zeros((n_coarse, ww, hh, dd)).astype(np.float32)
    score_map_f = np.zeros((n_fine, ww, hh, dd)).astype(np.float32)
    feat_map_f = np.zeros((param.network.base_feature_num, ww, hh, dd)).astype(np.float32)
    cnt = np.zeros((ww, hh, dd)).astype(np.float32)

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
                    
                yc = yc.cpu().data.numpy()[0]
                yf = yf.cpu().data.numpy()[0]
                feat = feat.cpu().data.numpy()[0]
                
                feat_map_f[:, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] = \
                    feat_map_f[:, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] + feat
                
                score_map_c[:, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] \
                    = score_map_c[:, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] + yc
                score_map_f[:, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] \
                    = score_map_f[:, xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] + yf
                cnt[xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] \
                    = cnt[xs, ys:ys+patch_size[0], zs:zs+patch_size[1]] + 1
                
    score_map_c = score_map_c/np.expand_dims(cnt, axis=0)
    score_map_f = score_map_f/np.expand_dims(cnt, axis=0)
    feat_map_f = feat_map_f / np.expand_dims(cnt, axis=0)
    label_map_c = np.argmax(score_map_c, axis=0)
    label_map_f = np.argmax(score_map_f, axis=0)
    
    if add_pad:
        label_map_c = label_map_c[:, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        label_map_f = label_map_f[:, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map_c = score_map_c[:, :, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map_f = score_map_f[:, :, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        feat_map_f = feat_map_f[:, :, hl_pad:hl_pad+h, dl_pad:dl_pad+d]

    return label_map_c, label_map_f, feat_map_f



def test_single_case_3d(net, image, stride_xy, stride_z, patch_size, n_labels):
    n_coarse, n_fine = n_labels
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
    _, _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map_c = np.zeros((n_coarse, ww, hh, dd)).astype(np.float32)
    score_map_f = np.zeros((n_fine, ww, hh, dd)).astype(np.float32)
    feat_map_f = np.zeros((param.network.base_feature_num, ww, hh, dd)).astype(np.float32)
    cnt = np.zeros((ww, hh, dd)).astype(np.float32)

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
                    
                yc = yc.cpu().data.numpy()[0]
                yf = yf.cpu().data.numpy()[0]
                feat = feat.cpu().data.numpy()[0]
                
                feat_map_f[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = \
                    feat_map_f[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + feat
                
                score_map_c[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map_c[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + yc
                score_map_f[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map_f[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + yf
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
                
    score_map_c = score_map_c/np.expand_dims(cnt, axis=0)
    score_map_f = score_map_f/np.expand_dims(cnt, axis=0)
    feat_map_f = feat_map_f / np.expand_dims(cnt, axis=0)
    label_map_c = np.argmax(score_map_c, axis=0)
    label_map_f = np.argmax(score_map_f, axis=0)
    
    if add_pad:
        label_map_c = label_map_c[wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        label_map_f = label_map_f[wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map_c = score_map_c[:, wl_pad:wl_pad + w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map_f = score_map_f[:, wl_pad:wl_pad + w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        feat_map_f = feat_map_f[:, wl_pad:wl_pad + w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]

    return label_map_c, label_map_f, feat_map_f


def test_all_case(net, param, testloader, gpu_id, stride_xy=32, stride_z=24, draw_ddm_im=False):

    total_metric_c = np.zeros((param.dataset.n_coarse - 1, n_eval))
    total_metric_f = np.zeros((param.dataset.n_fine, n_eval))
    n_images = len(testloader)

    tsne_index = 0 if draw_ddm_im else -1
    print("Testing begin")
    with open(os.path.join(param.path.path_to_dataset, 'test.list'), 'r') as fp:
        image_ids = fp.readlines()
    
    with open(os.path.join(param.path.path_to_test, 'test_log.txt'), "a") as fp:
        fp.writelines('test instance\t' + '\t'.join([method.__name__ for method in eval_metrics]) + '\n')
        
        for case_index, sampled_batch in enumerate(tqdm(testloader)):
            
            ids = image_ids[case_index].strip()
            metric_c, metric_f, feat_map = test_single_case(
                net, sampled_batch, stride_xy, stride_z, gpu_id=gpu_id, save_pred=False, ids=ids
            )
            
            if case_index == tsne_index:
                if not visualize(
                    feat_map, sampled_batch['fine'][0], 0, 'tsne', param,
                    os.path.join(param.path.path_to_test, f'tsne_{param.exp.exp_name}.eps'),
                    legend=param.dataset.legend, n_components=2
                ):
                    tsne_index += 1
            
            for c in range(param.dataset.n_coarse - 1):
                total_metric_c[c] += metric_c[c]
                fp.writelines(f'{ids}\t' + '\t'.join([str(round(metric_c[c, k], 5)) for k in range(n_eval)]))
                
            for f in range(param.dataset.n_fine - 1):
                total_metric_f[f] += metric_f[f]
                fp.writelines(f'{ids}\t' + '\t'.join([str(round(metric_f[f, k], 5)) for k in range(n_eval)]))
            print(f'avg fine for {ids}\t' + '\t'.join([str(round(metric_f[-1, k], 5)) for k in range(n_eval)]))
            fp.writelines(f'avg fine for {ids}\t' + '\t'.join([str(round(metric_f[-1, k], 5)) for k in range(n_eval)]))

        log = []
        for i in range(1, param.dataset.n_coarse):
            log.append(f'mean of superclass f{i}' + ','.join([f'{total_metric_c[i-1, j] / n_images}' for j in range(len(eval_metrics))]))
        fp.writelines('\n'.join(log))
        
        log = []
        for i in range(1, param.dataset.n_fine):
            log.append(f'mean of subclass f{i}' + ','.join([f'{total_metric_f[i-1, j] / n_images}' for j in range(len(eval_metrics))]))
        fp.writelines('\n'.join(log))
        
        mean_f = [total_metric_f[:, i].sum() / (param.dataset.n_fine - 1) for i in range(len(eval_metrics))]
        fp.writelines(f'mean of subclasses:' + ','.join([f'{i / n_images}' for i in mean_f]) + '\n')
        
    fp.close()
    print("Testing end")
    total_metric_f[-1] = mean_f
    return total_metric_c / n_images, total_metric_f / n_images


def calculate_metric_percase(pred, gt):
    ret = np.zeros((len(eval_metrics),))
    if pred.sum() > 0 and gt.sum() > 0:
        for i, met in enumerate(eval_metrics):
            ret[i] = met(pred, gt)

    return ret


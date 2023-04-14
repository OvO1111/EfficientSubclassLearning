import os
import sys
import math
import json
import torch
import shutil
import logging
import argparse

import numpy as np
from tqdm import tqdm
from medpy import metric
import torch.nn.functional as F
from collections import namedtuple
from torch.utils.data import DataLoader

from networks.unet import UNet
from val import test_single_case
from utils.visualize import visualize
from dataloaders.base_dataset import BaseDataset

eval_metrics = [metric.binary.dc, metric.binary.hd95, metric.binary.precision, metric.binary.recall]
n_eval = len(eval_metrics)
param = None


def test_all_case(net, param, testloader, gpu_id, stride_xy=64, stride_z=64, draw_ddm_im=False):
    
    logging.info(param)
    
    if os.path.exists(os.path.join(param.path.path_to_test, "test_log.txt")):
        os.remove(os.path.join(param.path.path_to_test, "test_log.txt"))

    total_metric_c = np.zeros((param.dataset.n_coarse - 1, n_eval))
    total_metric_f = np.zeros((param.dataset.n_fine, n_eval))
    all_total_metric_f = np.zeros((len(testloader), param.dataset.n_fine - 1, n_eval))
    n_images = len(testloader)

    tsne_index = 0 if draw_ddm_im else -1
    print("Testing begin")
    with open(os.path.join(param.path.path_to_dataset, 'test.list'), 'r') as fp:
        image_ids = fp.readlines()
    
    logging.info('test metrics:\t' + '\t'.join([method.__name__ for method in eval_metrics]) + '\n')
    
    for case_index, sampled_batch in enumerate(tqdm(testloader)):
        
        ids = image_ids[case_index].strip()
        metric_c, metric_f, feat_map = test_single_case(
            net, param, sampled_batch, stride_xy, stride_z, gpu_id=gpu_id, save_pred=False, ids=ids
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
            logging.debug(f'{ids}\t' + '\t'.join([f"{metric_c[c, k]:.3f}" for k in range(n_eval)]))
            
        for f in range(param.dataset.n_fine - 1):
            total_metric_f[f] += metric_f[f]
            all_total_metric_f[case_index] = metric_f[:-1]
            logging.debug(f'{ids}\t' + '\t'.join([f"{metric_f[f, k]:.3f}" for k in range(n_eval)]))
        logging.debug(f'avg fine for {ids}\t' + '\t'.join([f"{metric_f[-1, k]:.3f}" for k in range(n_eval)]))

    for i in range(1, param.dataset.n_coarse):
        log = f'mean of superclass {i}:\t' + '\t'.join([f"{_:.3f}" for _ in (total_metric_c[i-1] / n_images)])
        logging.info(log)
    
    for i in range(1, param.dataset.n_fine):
        log = f'mean of subclass {i}:\t' + '\t'.join([f"{_:.3f}" for _ in total_metric_f[i-1] / n_images])
        logging.info(log)
        log = f'std of subclass {i}:\t' + '\t'.join([f"{_:.3f}" for _ in np.std(all_total_metric_f[:, i-1], axis=0)])
        logging.info(log)
    
    mean_f = [total_metric_f[:, i].sum() / (param.dataset.n_fine - 1) for i in range(len(eval_metrics))]
    logging.info(f'mean of subclasses:\t' + '\t'.join([f'{i / n_images:.3f}' for i in mean_f]))
    # logging.info(f'std of all subclasses: {np.std(all_total_metric_f[:, i-1]):.5f}')

    total_metric_f[-1] = mean_f
    return total_metric_c / n_images, total_metric_f / n_images


def calculate_metric_percase(pred, gt):
    ret = np.zeros((len(eval_metrics),))
    if pred.sum() > 0 and gt.sum() > 0:
        for i, met in enumerate(eval_metrics):
            ret[i] = met(pred, gt)

    return ret


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['branched', 'unet'], default='unet', help='network type for selected trained model')
    parser.add_argument('-p', '--path', type=str, default='/nas/dailinrui/SSL4MIS/model_final/prostate/unet24', help='root dir of trained folder')
    parser.add_argument('-g', '--gpu', type=int, default=5, help='gpu on which to test model')
    args = parser.parse_args()
    
    with open(os.path.join(args.path, 'param.json'), 'r') as fp:
        d = json.load(fp)
    
    d1 = d['dataset']
    d2 = d['exp']
    d3 = d['path']
    d4 = d['network']
    P = namedtuple('P', ['dataset', 'exp', 'path', 'network'])
    param = P(dataset=namedtuple('dataset', d1.keys())(*d1.values()),
              exp=namedtuple('exp', d2.keys())(*d2.values()),
              path=namedtuple('path', d3.keys())(*d3.values()),
              network=namedtuple('network', d4.keys())(*d4.values()))

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(levelname)-5s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(os.path.join(param.path.path_to_test, "test_log.txt"), mode='w'),
                  logging.StreamHandler(sys.stdout)]
    )
    
    num_classes = (param.dataset.n_coarse, param.dataset.n_fine)
    test_save_path = param.path.path_to_test
    
    if args.model == 'branched':
        if param.dataset.n_coarse > 2:
            from networks.multiplebranchedunet import UNetMultiBranchNetwork
            net = UNetMultiBranchNetwork(param).cuda(args.gpu)
        elif param.dataset.n_coarse == 2:
            from networks.singlybranchedunet import UNetSingleBranchNetwork
            net = UNetSingleBranchNetwork(param).cuda(args.gpu)
    elif args.model == 'unet':
        net = UNet(param).cuda(args.gpu)
    
    save_mode_path = os.path.join(param.path.path_to_model, '{}_best_model.pth'.format(param.exp.exp_name))
    state_dicts = torch.load(save_mode_path, map_location='cpu')
    net.load_state_dict(state_dicts['model_state_dict'])
    print("init weight from {}".format(save_mode_path))
    net.eval()
    
    db_test = BaseDataset(param, split='test')
    testloader = DataLoader(db_test, num_workers=1, batch_size=1)
    
    test_all_case(net, param, testloader, stride_xy=64, stride_z=64, gpu_id=args.gpu)

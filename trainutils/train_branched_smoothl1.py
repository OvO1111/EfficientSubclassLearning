#!usr/bin/env python

import os
import sys
import math
import random
import shutil
import logging
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from test import test_all_case
from val import test_single_case
from utils import losses
from utils.parser import OmegaParser
from networks.singlybranchedunet import UNetSingleBranchNetwork
from utils.visualize import make_curve, make_image
from dataloaders.utils import TwoStreamBatchSampler


def train_c2f(model, optimizer, param: OmegaParser):
    model = model[0]
    optimizer = optimizer[0]
    
    base_lr = param.lr
    best_performance = 0.0
    iter_num = 0
    loss = {}
        
    batch_size = param.bs
    max_iterations = param.itr

    db_train = param.get_dataset(split='train')
    db_val = param.get_dataset(split='val')
    
    labeled_idxs = db_train.labeled_idxs
    unlabeled_idxs = db_train.unlabeled_idxs
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-param.labeled_bs)
    
    def worker_init_fn(worker_id):
        random.seed(param.seed + worker_id)

    trainloader = DataLoader(db_train,
                             num_workers=4,
                             pin_memory=True,
                             batch_sampler=batch_sampler,
                             worker_init_fn=worker_init_fn)
    
    valloader = DataLoader(db_val, num_workers=1, batch_size=1)

    model.train()
    
    ce_loss = torch.nn.CrossEntropyLoss()
    l1_loss = torch.nn.SmoothL1Loss()
    nce_loss = losses.NegativeCrossEntropyLoss()
    dice_loss_coarse = losses.DiceLoss(param.n_coarse)
    dice_loss_fine = losses.DiceLoss(param.n_fine) 

    writer = SummaryWriter(os.path.join(param.path_to_snapshot, "log"))
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = (max_iterations - iter_num) // (len(trainloader))
    iterator = tqdm(range(max_epoch), ncols=100, desc=f'{param.exp_name} Training Progress')
    torch.autograd.set_detect_anomaly(True)
    
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            q_im, q_lc, q_lf = sampled_batch['image'], sampled_batch['coarse'], sampled_batch['fine']
            q_im, q_lc, q_lf = q_im.cuda(), q_lc.cuda(), q_lf.cuda()

            out = model(q_im)
            out_coarse, out_fine = out['coarse_logit'], out['fine_logit']
            
            soft_coarse = torch.softmax(out_coarse, dim=1)
            soft_fine = torch.softmax(out_fine, dim=1)
            
            pred_coarse = torch.argmax(soft_coarse, dim=1)
            pred_fine = torch.round(out["mapped_cls"]).squeeze()
            
            loss_ce1 = ce_loss(out_coarse, q_lc)
            loss_dice1 = dice_loss_coarse(soft_coarse, q_lc)
            loss_coarse = 0.5 * (loss_ce1 + loss_dice1)
            loss_l1_2 = l1_loss(pred_fine[:param.labeled_bs], q_lf[:param.labeled_bs])
            # dice loss fine
            inputs = pred_fine[:param.labeled_bs]
            target = q_lf[:param.labeled_bs]
            smooth = 1e-8
            intersect = torch.sum(inputs * target)
            y_sum = torch.sum(target * target)
            z_sum = torch.sum(inputs * inputs)
            loss_dice2 = 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
            loss_fine = 0.5 * (loss_l1_2 + loss_dice2)
            
            loss['supervise loss coarse'] = loss_coarse
            loss['supervise loss fine'] = loss_fine
            
            if param.nl:
                loss['negative learning loss'] = nce_loss(out[param.labeled_bs:], q_lc[param.labeled_bs:])
            
            if param.d_h:
                
                mixed_im, mixed_lf, alpha = sampled_batch['mixed'], sampled_batch['fine'], sampled_batch['alpha']
                mixed_im, mixed_lf, alpha = mixed_im.cuda(), mixed_lf.cuda(), alpha.cuda()

                mixed_pred, pseudo_lf, mapped_cls = model.gen_mixup_labels(
                    q_im=q_im[param.labeled_bs:],
                    q_lc=q_lc[param.labeled_bs:],
                    q_soft=soft_fine[param.labeled_bs:],
                    mixed_im=mixed_im[param.labeled_bs:],
                    mixed_lf=mixed_lf[param.labeled_bs:],
                    alpha=alpha[param.labeled_bs:],
                    threshold=max(0.999 ** (iter_num // 10), 0.4),
                    with_pseudo_label=param.d_p
                )
                
                mapped_cls = torch.round(mapped_cls).squeeze()
                pseudo_lf = torch.argmax(pseudo_lf, dim=1)
                loss_l1_3 = l1_loss(mapped_cls, pseudo_lf)
                inputs = mapped_cls
                target = pseudo_lf
                mask = pseudo_lf > 0
                smooth = 1e-8
                intersect = torch.sum(inputs * target * mask)
                y_sum = torch.sum(target * target * mask)
                z_sum = torch.sum(inputs * inputs * mask)
                loss_dice3 = 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
                loss3 = 0.5 * (loss_dice3 + loss_l1_3) / (1 + math.exp(-iter_num // 1000))
                loss['sematic mixup loss'] = loss3
                
            elif param.d_p:
                
                pseudo_lf = model.gen_pseudo_labels(
                    q_im=q_im[param.labeled_bs:],
                    q_soft=soft_fine[param.labeled_bs:],
                    q_lc=q_lc[param.labeled_bs:],
                    threshold=max(0.999 ** (iter_num // 10), 0.4)
                )
                
                loss_ce3 = ce_loss(out_fine[param.labeled_bs:], pseudo_lf)
                loss_dice3 = dice_loss_fine(
                    soft_fine[param.labeled_bs:],
                    pseudo_lf, mask=pseudo_lf
                )
                loss3 = 0.5 * (loss_dice3 + loss_ce3) / (1 + math.exp(-iter_num // 1000))
                loss['pseudo label loss'] = loss3
            
            make_curve(writer, pred_fine, q_lf, 'train', param.n_fine, iter_num)
            make_curve(writer, pred_coarse, q_lc, 'train', param.n_coarse, iter_num)

            loss_sum = sum(loss.values())          
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar(f'{param.exp_name}/lr', lr_, iter_num)
            writer.add_scalar('loss/total_loss', loss_sum, iter_num)
            writer.add_scalars('loss/individual_losses', loss, iter_num)

            loss_names = list(loss.keys())
            loss_values = list(map(lambda x: str(round(x.item(), 3)), loss.values()))
            loss_log = ['*'] * (2 * len(loss_names))
            loss_log[::2] = loss_names
            loss_log[1::2] = loss_values
            loss_log = '; '.join(loss_log)
            logging.info(f"model {param.exp_name} iteration {iter_num} : total loss: {loss_sum.item():.3f},\n" + loss_log)

            if iter_num % 50 == 0:
                make_image(writer, param, q_im, 'image/input_image', iter_num, normalize=True)
                make_image(writer, param, q_lc, 'image/coarse_gt', iter_num, param.n_coarse - 1)
                make_image(writer, param, q_lf, 'image/fine_gt', iter_num, param.n_fine - 1)
                make_image(writer, param, pred_coarse, 'image/coarse_pred', iter_num, param.n_coarse - 1)
                make_image(writer, param, pred_fine, 'image/fine_pred', iter_num, param.n_fine - 1)
                
                if param.d_h:
                    make_image(writer, param, mixed_im, 'pseudo_label/mixup_image', iter_num, normalize=True, imindex=param.labeled_bs)
                    make_image(writer, param, mixed_lf, 'pseudo_label/mixup_fine_gt', iter_num, param.n_fine - 1, imindex=param.labeled_bs)
                if param.d_p:
                    make_image(writer, param, pseudo_lf, 'pseudo_label/pseudo_fine_gt', iter_num, param.n_fine - 1, imindex=0)  # imindex=param.labeled_bs)

            if iter_num > 0 and iter_num % param.val_step == 0:
                model.eval()
                avg_metric_f = np.zeros((len(valloader), param.n_fine, 4))
                for case_index, sampled_batch in enumerate(tqdm(valloader, position=1, leave=True, desc='Validation Progress')):
                    _, batch_metric_f, _ = test_single_case(model, param, sampled_batch, stride_xy=round(param.ps[0] * 0.7), stride_z=64)
                    avg_metric_f[case_index] = batch_metric_f
                
                if avg_metric_f[:, -1, param.eval_metric].mean() > best_performance:
                    best_performance = avg_metric_f[:, -1, param.eval_metric].mean()
                    save_best = os.path.join(param.path_to_model, '{}_best_model.pth'.format(param.exp_name))
                    torch.save({"model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "iterations": iter_num, "metric": best_performance}, save_best)
                    logging.info(f"save model to {save_best}")
                
                for index, name in enumerate(['dice', 'hd95', 'precision', 'recall']):
                    writer.add_scalars(f'val/{name}', {f'fine label={i}': avg_metric_f[:, i-1, index].mean() for i in range(1, param.n_fine)}, iter_num)
                    writer.add_scalars(f'val/{name}', {f'fine avg': avg_metric_f[:, -1, index].mean()}, iter_num)

                logging.info(f'iteration {iter_num} : dice_score : {avg_metric_f[:, -1, 0].mean():.4f} hd95 : {avg_metric_f[:, -1, 1].mean():.4f}')
                model.train()

            if iter_num > 0 and iter_num % param.val_step == 0:
                save_model_path = os.path.join(param.path_to_model, 'iter_' + str(iter_num) + '.pth')
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "iterations": iter_num, "metric": best_performance}, save_model_path)
                logging.info(f"save model to {save_model_path}")

            if iter_num == max_iterations:
                save_model_path = os.path.join(param.path_to_model, '{}_last_model.pth'.format(param.exp_name))
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "iterations": iter_num, "metric": best_performance}, save_model_path)
                logging.info(f"save model to {save_model_path}")
    
        if iter_num >= max_iterations:
            iterator.close()
            break
            
    writer.close()
    return "Training Finished!"


def test(model, parameter: OmegaParser):
    
    save_model_path = os.path.join(parameter.path_to_model, '{}_best_model.pth'.format(parameter.exp_name))
    model.load_state_dict(torch.load(save_model_path)['model_state_dict'])
    print("init weight from {}".format(save_model_path))
    
    db_test = parameter.get_dataset(split='test')
    testloader = DataLoader(db_test, num_workers=1, batch_size=1)
    
    model.eval()
    avg_metric_c, avg_metric_f =\
        test_all_case(model, parameter, testloader, stride_xy=64, stride_z=64)
    
    print(avg_metric_c)
    print(avg_metric_f)


if __name__ == "__main__":
    pass
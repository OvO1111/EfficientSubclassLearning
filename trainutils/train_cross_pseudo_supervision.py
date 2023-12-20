import os
import sys
import math
import random
import shutil
import logging
import numpy as np
from tqdm.auto import tqdm

import torch
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from val import test_single_case
from utils import losses
from utils.ramps import sigmoid_rampup
from utils.visualize import make_curve, make_image
from dataloaders.utils import TwoStreamBatchSampler


args = None


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train_cps(models, optimizers, param, parsed_args):
    global args
    args = parsed_args
    
    model1, model2 = models
    optimizer1, optimizer2 = optimizers
    
    base_lr = param.exp.base_lr
    batch_size = param.exp.batch_size
    max_iterations = param.exp.max_iter
    
    best_performance1 = 0.0
    best_performance2 = 0.0
    iter_num = 0
    loss = {}
    
    model1.train()
    model2.train()

    db_train = param.get_dataset(split='train')
    db_val = param.get_dataset(split='val')

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = db_train.labeled_idxs
    unlabeled_idxs = db_train.unlabeled_idxs
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-param.exp.labeled_batch_size)
    
    trainloader = DataLoader(db_train,
                             num_workers=4,
                             pin_memory=True,
                             batch_sampler=batch_sampler,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, num_workers=args.val_bs, batch_size=args.val_bs)

    ce_loss = CrossEntropyLoss()
    nce_loss = losses.NegativeCrossEntropyLoss()
    dice_loss_coarse = losses.DiceLoss(param.dataset.n_coarse)
    dice_loss_fine = losses.DiceLoss(param.dataset.n_fine) 

    writer = SummaryWriter(os.path.join(param.path.path_to_snapshot, "log"))
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = (max_iterations - iter_num) // (len(trainloader))
    iterator = tqdm(range(max_epoch), ncols=100, desc=f'{args.exp_name} Training Progress')
    torch.autograd.set_detect_anomaly(True)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            q_im, q_lc, q_lf = sampled_batch['image'], sampled_batch['coarse'], sampled_batch['fine']

            if args.gpu >= 0:
                q_im, q_lc, q_lf = q_im.cuda(args.gpu), q_lc.cuda(args.gpu), q_lf.cuda(args.gpu)
            else: raise RuntimeError(f'Specify a positive gpu id')

            out1 = model1(q_im)
            out_coarse1, out_fine1 = out1['coarse_logit'], out1['fine_logit']
            soft_coarse1, soft_fine1 = torch.softmax(out_coarse1, dim=1), torch.softmax(out_fine1, dim=1)
            pred_fine1 = torch.argmax(soft_fine1, dim=1)

            out2 = model2(q_im)
            out_coarse2, out_fine2 = out2['coarse_logit'], out2['fine_logit']
            soft_coarse2, soft_fine2 = torch.softmax(out_coarse2, dim=1), torch.softmax(out_fine2, dim=1)
            pred_fine2 = torch.argmax(soft_fine2, dim=1)
            
            make_curve(writer, pred_fine1, q_lf, 'model1', param.dataset.n_fine, iter_num)
            make_curve(writer, pred_fine2, q_lf, 'model2', param.dataset.n_fine, iter_num)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss['model1 supervise loss'] = 0.25 * (ce_loss(out_coarse1, q_lc) + dice_loss_coarse(soft_coarse1, q_lc) + \
                    ce_loss(out_fine1[:args.labeled_bs], q_lf[:args.labeled_bs]) + dice_loss_fine(soft_fine1[:args.labeled_bs], q_lf[:args.labeled_bs]))
            
            loss['model2 supervise loss'] = 0.25 * (ce_loss(out_coarse2, q_lc) + dice_loss_coarse(soft_coarse2, q_lc) + \
                    ce_loss(out_fine2[:args.labeled_bs], q_lf[:args.labeled_bs]) + dice_loss_fine(soft_fine2[:args.labeled_bs], q_lf[:args.labeled_bs]))

            pseudo_outputs_f1 = torch.argmax(soft_fine1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs_f2 = torch.argmax(soft_fine2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision_f1 = ce_loss(out_fine1[args.labeled_bs:], pseudo_outputs_f2)
            pseudo_supervision_f2 = ce_loss(out_fine2[args.labeled_bs:], pseudo_outputs_f1)

            loss['model1 supervise loss'] += consistency_weight * (pseudo_supervision_f1)
            loss['model2 supervise loss'] += consistency_weight * (pseudo_supervision_f2)

            loss_sum = sum(loss.values())

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss_sum.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer2.param_groups:
                param_group2['lr'] = lr_

            if args.verbose:
                loss_names = list(loss.keys())
                loss_values = list(map(lambda x: str(round(x.item(), 3)), loss.values()))
                loss_log = ['*'] * (2 * len(loss_names))
                loss_log[::2] = loss_names
                loss_log[1::2] = loss_values
                loss_log = '; '.join(loss_log)
                logging.info(f"model {param.exp.exp_name} iteration {iter_num} : total loss: {loss_sum.item():.3f},\n" + loss_log)
            
            if iter_num % args.draw_step == 0:
                make_image(writer, param, q_im, 'image/input_image', iter_num, normalize=True)
                make_image(writer, param, q_lf, 'image/fine_gt', iter_num, param.dataset.n_fine - 1)
                make_image(writer, param, pred_fine1, 'image/model1_fine_pred', iter_num, param.dataset.n_fine - 1)
                make_image(writer, param, pred_fine2, 'image/model2_fine_pred', iter_num, param.dataset.n_fine - 1)

            if iter_num > 0 and iter_num % args.val_step == 0:
                model1.eval()
                model2.eval()
                avg_metric_f1 = np.zeros((len(valloader), param.dataset.n_fine, 4))
                avg_metric_f2 = np.zeros((len(valloader), param.dataset.n_fine, 4))
                for case_index, sampled_batch in enumerate(tqdm(valloader, position=1, leave=True, desc='Validation Progress')):
                    _, batch_metric_f1, _ = test_single_case(model1, param, sampled_batch, stride_xy=round(param.exp.patch_size[0] * 0.7), stride_z=64, gpu_id=args.gpu)
                    avg_metric_f1[case_index] = batch_metric_f1
                for case_index, sampled_batch in enumerate(tqdm(valloader, position=1, leave=True, desc='Validation Progress')):
                    _, batch_metric_f2, _ = test_single_case(model2, param, sampled_batch, stride_xy=round(param.exp.patch_size[0] * 0.7), stride_z=64, gpu_id=args.gpu)
                    avg_metric_f2[case_index] = batch_metric_f2
                
                save_best_path = os.path.join(param.path.path_to_model, '{}_best_model.pth'.format(param.exp.exp_name))
                last_best_model1_state_dict = None
                last_best_model2_state_dict = None
                if os.path.exists(save_best_path):
                    state_dicts = torch.load(save_best_path, map_location='cpu')
                    last_best_model1_state_dict = state_dicts['model_state_dict']
                    last_best_model2_state_dict = state_dicts['model2_state_dict']
                    
                this_performance1 = avg_metric_f1[:, -1, param.exp.eval_metric].mean()
                this_performance2 = avg_metric_f2[:, -1, param.exp.eval_metric].mean()
                if this_performance1 > best_performance1 or this_performance2 > best_performance2:
                    if this_performance1 > best_performance1 or last_best_model1_state_dict is None:
                        best_model1_state_dict = model1.state_dict()
                        best_performance1 = avg_metric_f1[:, -1, param.exp.eval_metric].mean()
                    else:
                        best_model1_state_dict = last_best_model1_state_dict
                    if this_performance2 > best_performance2 or last_best_model2_state_dict is None:
                        best_model2_state_dict = model2.state_dict()
                        best_performance2 = avg_metric_f2[:, -1, param.exp.eval_metric].mean()
                    else:
                        best_model2_state_dict = last_best_model2_state_dict
                    
                    torch.save({"model_state_dict": best_model1_state_dict,
                                "model2_state_dict": best_model2_state_dict,
                                "optimizer_state_dict": optimizer1.state_dict(),
                                "optimizer2_state_dict": optimizer2.state_dict(),
                                "iterations": iter_num, "metric": best_performance1, "metric2": best_performance2}, save_best_path)
                    logging.info(f"save model to {save_best_path}")
                
                for index, name in enumerate(['dsc', 'hd95', 'precision', 'recall']):
                    writer.add_scalars(f'val/model1_{name}', {f'fine label={i}': avg_metric_f1[:, i-1, index].mean() for i in range(1, param.dataset.n_fine)}, iter_num)
                    writer.add_scalars(f'val/model1_{name}', {f'fine avg': avg_metric_f1[:, -1, index].mean()}, iter_num)
                    writer.add_scalars(f'val/model2_{name}', {f'fine label={i}': avg_metric_f2[:, i-1, index].mean() for i in range(1, param.dataset.n_fine)}, iter_num)
                    writer.add_scalars(f'val/model2_{name}', {f'fine avg': avg_metric_f2[:, -1, index].mean()}, iter_num)

                logging.info(f'iteration {iter_num} : [model 1] dice_score : {avg_metric_f1[:, -1, 0].mean():.4f}; hd95 : {avg_metric_f1[:, -1, 1].mean():.4f}')
                logging.info(f'iteration {iter_num} : [model 2] dice_score : {avg_metric_f2[:, -1, 0].mean():.4f}; hd95 : {avg_metric_f2[:, -1, 1].mean():.4f}')
                model1.train()
                model2.train()

            if iter_num > 0 and iter_num % args.save_step == 0:
                save_model_path = os.path.join(param.path.path_to_model, 'iter_' + str(iter_num) + '.pth')
                torch.save({"model_state_dict": model1.state_dict(),
                            "model2_state_dict": model2.state_dict(),
                            "optimizer_state_dict": optimizer1.state_dict(),
                            "optimizer2_state_dict": optimizer2.state_dict(),
                            "iterations": iter_num, "loss": loss_sum.item()}, save_model_path)
                logging.info(f"save model to {save_model_path}")

            if iter_num >= max_iterations:
                save_model_path = os.path.join(param.path.path_to_model, '{}_last_model.pth'.format(param.exp.exp_name))
                torch.save({"model_state_dict": model1.state_dict(),
                            "model2_state_dict": model2.state_dict(),
                            "optimizer_state_dict": optimizer1.state_dict(),
                            "optimizer2_state_dict": optimizer2.state_dict(),
                            "iterations": iter_num, "loss": loss_sum.item()}, save_model_path)
                logging.info(f"save model to {save_model_path}")
            
        if iter_num >= max_iterations:
            iterator.close()
            break
        
    writer.close()
    return "Training Finished!"
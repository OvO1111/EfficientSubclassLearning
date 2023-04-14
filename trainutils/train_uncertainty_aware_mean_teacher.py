import os
import math
import random
import shutil
import logging
import numpy as np
from tqdm.auto import tqdm

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from test import test_all_case
from val import test_single_case
from utils import losses
from utils.ramps import sigmoid_rampup
from utils.visualize import make_curve, make_image
from dataloaders.utils import TwoStreamBatchSampler


param = None
args = None


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train_uamt(models, optimizer, parameter, parsed_args):
    global param, args
    param = parameter
    args = parsed_args
    print(models, optimizer, parameter, parsed_args)
    
    model, ema_model = models
    optimizer = optimizer[0]
    
    base_lr = param.exp.base_lr
    batch_size = param.exp.batch_size
    max_iterations = param.exp.max_iter
    
    best_performance = 0.0
    iter_num = 0
    loss = {}

    db_train = param.get_dataset(split='train')
    db_val = param.get_dataset(split='val')

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = db_train.labeled_idxs
    unlabeled_idxs = db_train.unlabeled_idxs
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-parameter.exp.labeled_batch_size)

    trainloader = DataLoader(db_train,
                             num_workers=4,
                             pin_memory=True,
                             batch_sampler=batch_sampler,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, num_workers=args.val_bs, batch_size=args.val_bs)

    ce_loss = CrossEntropyLoss()
    nce_loss = losses.NegativeCrossEntropyLoss()
    dice_loss_coarse = losses.DiceLoss(parameter.dataset.n_coarse)
    dice_loss_fine = losses.DiceLoss(parameter.dataset.n_fine) 

    writer = SummaryWriter(os.path.join(parameter.path.path_to_snapshot, "log"))
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = (max_iterations - iter_num) // (len(trainloader))
    iterator = tqdm(range(max_epoch), ncols=100, position=0, leave=True, desc='Training Progress')
    torch.autograd.set_detect_anomaly(True)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            q_im, q_lc, q_lf = sampled_batch['image'], sampled_batch['coarse'], sampled_batch['fine']

            if args.gpu >= 0:
                q_im, q_lc, q_lf = q_im.cuda(args.gpu), q_lc.cuda(args.gpu), q_lf.cuda(args.gpu)
            else: raise RuntimeError(f'Specify a positive gpu id')
            unlabeled_volume_batch = q_im[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            out = model(q_im)
            out_coarse, out_fine = out['coarse_logit'], out['fine_logit']
            soft_coarse, soft_fine = torch.softmax(out_coarse, dim=1), torch.softmax(out_fine, dim=1)
            pred_fine = torch.argmax(soft_fine, dim=1)
            
            with torch.no_grad():
                ema_out = ema_model(ema_inputs)
                ema_out_coarse, ema_out_fine = ema_out['coarse_logit'], ema_out['fine_logit']
            T = 8
            
            if param.dataset.n_dim == 3:
                _, _, d, w, h = unlabeled_volume_batch.shape
                volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2
                preds_f = torch.zeros([stride * T, param.dataset.n_fine, d, w, h], device=q_im.device)
                
                for i in range(T // 2):
                    ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                        ema_out2 = ema_model(ema_inputs)
                        preds_f[2 * stride * i: 2 * stride * (i + 1)] = ema_out2['fine_logit']
                
                preds_f = torch.softmax(preds_f, dim=1)
                preds_f = preds_f.reshape(T, stride, param.dataset.n_fine, d, w, h)
                preds_f = torch.mean(preds_f, dim=0)
                uncertainty_f = -1.0 * torch.sum(preds_f * torch.log(preds_f + 1e-6), dim=1, keepdim=True)
                
            elif math.floor(param.dataset.n_dim) == 2:
                _, _, w, h = unlabeled_volume_batch.shape
                volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2
                preds_f = torch.zeros([stride * T, param.dataset.n_fine, w, h], device=q_im.device)
                
                for i in range(T // 2):
                    ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                        ema_out2 = ema_model(ema_inputs)
                        preds_f[2 * stride * i: 2 * stride * (i + 1)] = ema_out2['fine_logit']
                        
                preds_f = torch.softmax(preds_f, dim=1)
                preds_f = preds_f.reshape(T, stride, param.dataset.n_fine, w, h)
                preds_f = torch.mean(preds_f, dim=0)
                uncertainty_f = -1.0 * torch.sum(preds_f * torch.log(preds_f + 1e-6), dim=1, keepdim=True)

            loss_ce1 = ce_loss(out_coarse, q_lc)
            loss_ce2 = ce_loss(out_fine[:args.labeled_bs], q_lf[:args.labeled_bs])
            loss_dice1 = dice_loss_coarse(soft_coarse, q_lc)
            loss_dice2 = dice_loss_fine(soft_fine[:args.labeled_bs], q_lf[:args.labeled_bs])
            loss['supervised loss coarse'] = 0.5 * (loss_dice1 + loss_ce1)
            loss['supervised loss fine'] = 0.5 * (loss_dice2 + loss_ce2)

            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist_f = losses.softmax_mse_loss(out_fine[args.labeled_bs:], ema_out_fine)
            threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask_f = (uncertainty_f < threshold).float()
            consistency_loss = torch.sum(mask_f * consistency_dist_f) / (2 * torch.sum(mask_f) + 1e-16)

            loss['consistency loss'] = consistency_weight * consistency_loss
            
            loss_sum = sum(loss.values())
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            if args.verbose:
                loss_names = list(loss.keys())
                loss_values = list(map(lambda x: str(round(x.item(), 3)), loss.values()))
                loss_log = ['*'] * (2 * len(loss_names))
                loss_log[::2] = loss_names
                loss_log[1::2] = loss_values
                loss_log = '; '.join(loss_log)
                logging.info(f"model {parameter.exp.exp_name} iteration {iter_num} : total loss: {loss_sum.item():.3f},\n" + loss_log)
            
            if iter_num % args.draw_step == 0:
                make_image(writer, parameter, q_im, 'image/input_image', iter_num, normalize=True)
                make_image(writer, parameter, q_lf, 'image/fine_gt', iter_num, parameter.dataset.n_fine - 1)
                make_image(writer, parameter, pred_fine, 'image/model1_fine_pred', iter_num, parameter.dataset.n_fine - 1)

            if iter_num > 0 and iter_num % args.val_step == 0:
                model.eval()
                avg_metric_f = np.zeros((len(valloader), parameter.dataset.n_fine, 4))
                for case_index, sampled_batch in enumerate(tqdm(valloader, position=1, leave=True, desc='Validation Progress')):
                    _, batch_metric_f, _ = test_single_case(model, parameter, sampled_batch, stride_xy=round(parameter.exp.patch_size[0] * 0.7), stride_z=64, gpu_id=args.gpu)
                    avg_metric_f[case_index] = batch_metric_f
                
                if avg_metric_f[:, -1, parameter.exp.eval_metric].mean() > best_performance:
                    best_performance = avg_metric_f[:, -1, parameter.exp.eval_metric].mean()
                    save_best = os.path.join(parameter.path.path_to_model, '{}_best_model.pth'.format(parameter.exp.exp_name))
                    torch.save({"model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "iterations": iter_num, "metric": best_performance}, save_best)
                    logging.info(f"save model to {save_best}")
                
                for index, name in enumerate(['dsc', 'hd95', 'precision', 'recall']):
                    writer.add_scalars(f'val/{name}', {f'fine label={i}': avg_metric_f[:, i-1, index].mean() for i in range(1, parameter.dataset.n_fine)}, iter_num)
                    writer.add_scalars(f'val/{name}', {f'fine avg': avg_metric_f[:, -1, index].mean()}, iter_num)

                logging.info(f'iteration {iter_num} : dice_score : {avg_metric_f[:, -1, 0].mean():.4f} hd95 : {avg_metric_f[:, -1, 1].mean():.4f}')
                model.train()

            if iter_num > 0 and iter_num % args.save_step == 0:
                save_model_path = os.path.join(parameter.path.path_to_model, 'iter_' + str(iter_num) + '.pth')
                torch.save({"model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "iterations": iter_num, "metric": best_performance}, save_model_path)
                logging.info(f"save model to {save_model_path}")

            if iter_num >= max_iterations:
                save_model_path = os.path.join(parameter.path.path_to_model, '{}_last_model.pth'.format(parameter.exp.exp_name))
                torch.save({"model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "iterations": iter_num, "metric": best_performance}, save_model_path)
                logging.info(f"save model to {save_model_path}")
            
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
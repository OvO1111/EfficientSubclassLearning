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

from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from utils import losses
from utils.parser import Parser
from networks.proposed import CU3D
from dataloaders.refuge2020 import Refuge2020
from test import test_single_case, test_all_case
from utils.visualize import make_curve, make_image
from dataloaders.utils import RandomRotFlip, RandomNoise, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
# hyper settings
parser.add_argument('-s', '--seed', type=int, default=1234, help='randomization seed')
parser.add_argument('-g', '--gpu', type=int, default=1, help='gpu on which to train model')

# experiment settings
parser.add_argument('--bs', type=int, default=24, help='number of batch size')
parser.add_argument('--lr', type=float, default=1e-2, help='base learning rate')
parser.add_argument('--iter', type=int, default=40000, help='maximum training iterations')
parser.add_argument('--eval', type=str, default='dsc', help='evaluation metric for saving model: [dsc, hd95, precision, recall]')
parser.add_argument('--m', action='store_true', help='whether to use label mixup')
parser.add_argument('--p', action='store_true', help='whether to use pseudo labeling')
parser.add_argument('--sn', action='store_true', help='whether to use separate batchnorm')
parser.add_argument('--pc', action='store_true', help='whether to use priority concatenation')
parser.add_argument('--restore', action='store_true', help='whether to continue a previous training')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='size for network input')
parser.add_argument('--exp_name', type=str, default='Fully_Supervised_140', help='name of the current model')

# path settings
parser.add_argument('--data_path', type=str, default='/data/dailinrui/dataset/refuge2020', help='root path for dataset')
parser.add_argument('--model_path', type=str, default='/nas/dailinrui/SSL4MIS/model_final/REFUGE2020', help='root path for training model')

# number of dataset samples for SSL
# for ACDC or any 3d database with a large interslice spacing and is trained per slice, this is the number of total slices
parser.add_argument('--labeled_bs', type=int, default=8, help='how many samples are labeled')
parser.add_argument('--total_num', type=int, default=1312, help='how many samples in total')
parser.add_argument('--labeled_num', type=int, default=1312, help='how many samples are labeled')

# network settings
parser.add_argument('--feature_scale', type=int, default=2, help='feature scale per unet encoder/decoder step')
parser.add_argument('--base_feature', type=int, default=32, help='base feature channels for unet layer 0')
parser.add_argument('--image_scale', type=int, default=2, help='image scale per unet encoder/decoder step')
parser.add_argument('--is_batchnorm', type=bool, default=True, help='use batchnorm instead of instancenorm')

# irrelevants
parser.add_argument('--val_step', type=int, default=200, help='do validation per val_step')
parser.add_argument('--draw_step', type=int, default=50, help='add train graphic result per draw_step')
parser.add_argument('--verbose', action='store_true', help='whether to display the loss information per iter')
args = parser.parse_args()
param = Parser(args).get_param()


def train(model, restore=False):
    list_trained_iterations = os.listdir(param.path.path_to_model)
    base_lr = param.exp.base_lr
    best_performance = 0.0
    iter_num = 0
    loss = {}
    
    if len(list_trained_iterations) == 0:
        restore = False
        
    if restore:
        list_trained_iterations.remove(f'{param.exp.exp_name}_best_model.pth')
        max_trained_iterations = list(map(lambda x: int(x.split('_')[1].rstrip('.pth')), list_trained_iterations))
        restore_itr = int(max_trained_iterations[-1])

        model_name = f'iter_{restore_itr}'
        for name in list_trained_iterations:
            if name.startswith(model_name):
                model_name = name
                break
        else:
            print('valid model not found')
            raise ValueError
        save_model_path = os.path.join(param.path.path_to_model, model_name)
        model.load_state_dict(torch.load(save_model_path, map_location='cpu'))
        best_performance = max([float(f.split('_')[-1].rstrip('.pth')) for f in list_trained_iterations if 'd' in f])
        print("init weight from {}, current best performance is {}".format(save_model_path, best_performance))
        base_lr = base_lr * (restore_itr / param.exp.max_iter)
        iter_num = restore_itr
        
    batch_size = param.exp.batch_size
    max_iterations = param.exp.max_iter
    
    labeled_idxs = list(range(param.exp.labeled_num))
    # labeled_idxs = random.sample(range(param.dataset.total_num), k=param.exp.labeled_num)
    unlabeled_idxs = list(range(param.dataset.total_num))
    for idx in labeled_idxs: unlabeled_idxs.remove(idx)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-param.exp.labeled_batch_size)

    db_train = param.get_dataset(param, split='train', labeled_idx=labeled_idxs)
    db_val = param.get_dataset(param, split='val')
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train,
                             num_workers=4,
                             pin_memory=True,
                             batch_sampler=batch_sampler,
                             worker_init_fn=worker_init_fn)
    
    valloader = DataLoader(db_val, num_workers=1, batch_size=1)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    ce_loss = CrossEntropyLoss()
    nce_loss = losses.NegativeCrossEntropyLoss()
    dice_loss_coarse = losses.DiceLoss(param.dataset.n_coarse)
    dice_loss_fine = losses.DiceLoss(param.dataset.n_fine) 

    writer = SummaryWriter(os.path.join(param.path.path_to_snapshot, "log"))
    logging.info("{} iterations per epoch".format(param.dataset.total_num // len(trainloader)))

    max_epoch = (max_iterations - iter_num) // (len(trainloader)) + 1
    iterator = tqdm(range(max_epoch), ncols=100, position=0, leave=True, desc='Training Progress')
    torch.autograd.set_detect_anomaly(True)
    
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            q_im, q_lc, q_lf = sampled_batch['image'], sampled_batch['coarse'], sampled_batch['fine']

            if args.gpu >= 0:
                q_im, q_lc, q_lf = q_im.cuda(args.gpu), q_lc.cuda(args.gpu), q_lf.cuda(args.gpu)
            else:
                raise RuntimeError(f'Specify a positive gpu id')

            out = model(q_im)
            out_coarse, out_fine = out['coarse_logit'], out['fine_logit']
            
            soft_coarse = torch.softmax(out_coarse, dim=1)
            soft_fine = torch.softmax(out_fine, dim=1)
            
            pred_coarse = torch.argmax(soft_coarse, dim=1)
            pred_fine = torch.argmax(soft_fine, dim=1)
            
            loss_ce1 = ce_loss(out_coarse, q_lc)
            loss_dice1 = dice_loss_coarse(soft_coarse, q_lc)
            loss_coarse = 0.5 * (loss_ce1 + loss_dice1)
            loss_ce2 = ce_loss(out_fine[:param.exp.labeled_batch_size], q_lf[:param.exp.labeled_batch_size])
            loss_dice2 = dice_loss_fine(soft_fine[:param.exp.labeled_batch_size], q_lf[:param.exp.labeled_batch_size])
            loss_fine = 0.5 * (loss_ce2 + loss_dice2)
            
            loss['supervise loss coarse'] = loss_coarse
            loss['supervise loss fine'] = loss_fine
            
            if param.exp.mixup_label:
                
                mixed_im, mixed_lf, alpha = sampled_batch['mixed'], sampled_batch['fine'], sampled_batch['alpha']
                if args.gpu > 0:
                    mixed_im, mixed_lf, alpha = mixed_im.cuda(args.gpu), mixed_lf.cuda(args.gpu), alpha.cuda(args.gpu)
                else:
                    raise RuntimeError(f'Specify a positive gpu id')

                mixed_pred, pseudo_lf = model.gen_mixup_labels(
                    q_im=q_im[param.exp.labeled_batch_size:],
                    q_lc=q_lc[param.exp.labeled_batch_size:],
                    q_soft=soft_fine[param.exp.labeled_batch_size:],
                    mixed_im=mixed_im[param.exp.labeled_batch_size:],
                    mixed_lf=mixed_lf[param.exp.labeled_batch_size:],
                    alpha=alpha[param.exp.labeled_batch_size:],
                    threshold=max(0.999 ** (iter_num // 10), 0.4),
                    with_pseudo_label=param.exp.pseudo_label
                )
                
                soft_mixed_pred = torch.softmax(mixed_pred, dim=1)
                loss_ce3 = ce_loss(mixed_pred, pseudo_lf)
                loss_dice3 = dice_loss_fine(soft_mixed_pred, pseudo_lf, mask=pseudo_lf)
                loss3 = 0.5 * (loss_dice3 + loss_ce3) / (1 + math.exp(-iter_num // 1000))
                loss['sematic mixup loss'] = loss3
                
            elif param.exp.pseudo_label:
                
                pseudo_lf = model.gen_pseudo_labels(
                    q_im=q_im[param.exp.labeled_batch_size:],
                    q_soft=soft_fine[param.exp.labeled_batch_size:],
                    q_lc=q_lc[param.exp.labeled_batch_size:],
                    threshold=max(0.999 ** (iter_num // 10), 0.4)
                )
                
                loss_ce3 = ce_loss(out_fine[param.exp.labeled_batch_size:], pseudo_lf)
                loss_dice3 = dice_loss_fine(
                    soft_fine[param.exp.labeled_batch_size:],
                    pseudo_lf, mask=pseudo_lf
                )
                loss3 = 0.5 * (loss_dice3 + loss_ce3) / (1 + math.exp(-iter_num // 1000))
                loss['pseudo label loss'] = loss3
            
            make_curve(writer, pred_fine, q_lf, 'train', param.dataset.n_fine, iter_num)
            make_curve(writer, pred_coarse, q_lc, 'train', param.dataset.n_coarse, iter_num)

            # loss4 = nce_loss(out_fine[param.exp.labeled_batch_size:], q_lc[param.exp.labeled_batch_size:])
            # loss['negative learning loss'] = loss4

            loss_sum = sum(loss.values())          
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar(f'{param.exp.exp_name}/lr', lr_, iter_num)
            writer.add_scalar('loss/total_loss', loss_sum, iter_num)
            writer.add_scalars('loss/individual_losses', loss, iter_num)

            if args.verbose:
                loss_names = list(loss.keys())
                loss_values = list(map(lambda x: str(round(x.item(), 5)), loss.values()))
                loss_log = ['*'] * (2 * len(loss_names))
                loss_log[::2] = loss_names
                loss_log[1::2] = loss_values
                loss_log = '; '.join(loss_log)
                logging.info(f"model {param.exp.exp_name} iteration {iter_num} : total loss: {loss_sum.item()},\n" + loss_log)

            if iter_num % args.draw_step == 0:
                make_image(writer, q_im, 'image/input_image', iter_num, normalize=True)
                make_image(writer, q_lc, 'image/coarse_gt', iter_num, param.dataset.n_coarse - 1)
                make_image(writer, q_lf, 'image/fine_gt', iter_num, param.dataset.n_fine - 1)
                make_image(writer, pred_coarse, 'image/coarse_pred', iter_num, param.dataset.n_coarse - 1)
                make_image(writer, pred_fine, 'image/fine_pred', iter_num, param.dataset.n_fine - 1)
                
                if param.exp.mixup_label or param.exp.pseudo_label:
                    make_image(writer, mixed_im, 'pseudo_label/mixup_image', iter_num, normalize=True)
                    make_image(writer, mixed_lf, 'pseudo_label/mixup_fine_gt', iter_num, param.dataset.n_fine - 1)

            if iter_num > 0 and iter_num % args.val_step == 0:
                model.eval()
                avg_metric_f = np.zeros((len(valloader), param.dataset.n_fine, 4))
                for case_index, sampled_batch in enumerate(tqdm(valloader, position=1, leave=True, desc='Validation Progress')):
                    _, batch_metric_f, _ = test_single_case(model, sampled_batch, stride_xy=round(param.exp.patch_size[0] * 0.7), stride_z=64, gpu_id=args.gpu)
                    avg_metric_f[case_index] = batch_metric_f
                
                if avg_metric_f[:, -1, param.exp.eval_metric].mean() > best_performance:
                    best_performance = avg_metric_f[:, -1, param.exp.eval_metric].mean()
                    save_model_path = os.path.join(param.path.path_to_model, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(param.path.path_to_model, '{}_best_model.pth'.format(param.exp.exp_name))
                    torch.save(model.state_dict(), save_model_path)
                    torch.save(model.state_dict(), save_best)
                
                for index, name in enumerate(['dsc', 'hd95', 'precision', 'recall']):
                    writer.add_scalars(f'val/{name}', {f'fine label={i}': avg_metric_f[:, i-1, index].mean() for i in range(1, param.dataset.n_fine)}, iter_num)
                    writer.add_scalars(f'val/{name}', {f'fine avg': avg_metric_f[:, -1, index].mean()}, iter_num)

                logging.info(f'iteration {iter_num} : dice_score : {avg_metric_f[:, -1, 0].mean():.5f} hd95 : {avg_metric_f[:, -1, 1].mean():.5f}')
                model.train()

            if iter_num % 5000 == 0:
                save_model_path = os.path.join(param.path.path_to_model, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_model_path)
                logging.info("save model to {}".format(save_model_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


def test(model):
    
    save_model_path = os.path.join(param.path.path_to_model, '{}_best_model.pth'.format(param.exp.exp_name))
    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))
    
    db_test = Refuge2020(param, split='test')
    testloader = DataLoader(db_test, num_workers=1, batch_size=1)
    
    model.eval()
    avg_metric_c, avg_metric_f =\
        test_all_case(model, param, testloader, stride_xy=64, stride_z=64, gpu_id=args.gpu)
    
    print(avg_metric_c)
    print(avg_metric_f)


if __name__ == "__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.basicConfig(
        filename=os.path.join(param.path.path_to_snapshot, "log.txt"),
        level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(msg=param)
    
    # model = unet_3D(in_channels=4).cuda(args.gpu)
    model = CU3D(in_channels=param.dataset.n_mode).cuda(args.gpu)
    
    train(model, restore=param.exp.restore)
    test(model)
    print(f'train-test over for {param.exp.exp_name}')
import os
import sys
import random
import shutil
import logging
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from test import test_all_case
from networks.unet import UNet
from utils.parser import Parser
from networks.singlybranchedunet import UNetSingleBranchNetwork
from networks.multiplebranchedunet import UNetMultiBranchNetwork

from networks.utils import init_weights
from trainutils.train_branched import train_c2f
from trainutils.train_plain_unet import train_unet
from trainutils.train_cross_pseudo_supervision import train_cps
from trainutils.train_uncertainty_aware_mean_teacher import train_uamt

parser = argparse.ArgumentParser()
# hyper settings
parser.add_argument('--seed', type=int, default=1234, help='randomization seed')
parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu on which to train model')

# experiment settings
parser.add_argument('--bs', type=int, default=24, help='number of batch size')
parser.add_argument('--lr', type=float, default=1e-2, help='base learning rate')
parser.add_argument('--iter', type=int, default=40000, help='maximum training iterations')
parser.add_argument('-m', '--mixup', action='store_true', help='whether to use label mixup')
parser.add_argument('-p', '--pseudo', action='store_true', help='whether to use pseudo labeling')
parser.add_argument('-s', '--sn', action='store_true', help='whether to use separate batchnorm')
parser.add_argument('-c', '--pc', action='store_true', help='whether to use priority concatenation')
parser.add_argument('--restore', action='store_true', help='whether to continue a previous training')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='size for network input')
parser.add_argument('--exp_name', type=str, default='mpsc_anycoarse', help='name of the current model')
parser.add_argument('--model', choices=['unet', 'cps', 'uamt', 'branched'], default='branched', help='which type of model to train')
parser.add_argument('--eval', type=str, default='dsc', choices=['dsc', 'hd95', 'precision', 'recall'], help='evaluation metric for saving model')

# path settings
parser.add_argument('--data_path', type=str, default='/data/dailinrui/dataset/prostate', help='root path for dataset')
parser.add_argument('--model_path', type=str, default='/nas/dailinrui/SSL4MIS/model_final/prostate', help='root path for training model')

# number of dataset samples for SSL
# for ACDC or any 3d database with a large interslice spacing and is trained per slice, this is the number of total slices
parser.add_argument('--labeled_bs', type=int, default=4, help='how many samples are labeled')
parser.add_argument('--total_num', type=int, default=458, help='how many samples in total')
parser.add_argument('--labeled_num', type=int, default=77, help='how many samples are labeled')

# network settings
parser.add_argument('--feature_scale', type=int, default=2, help='feature scale per unet encoder/decoder step')
parser.add_argument('--base_feature', type=int, default=16, help='base feature channels for unet layer 0')
parser.add_argument('--image_scale', type=int, default=2, help='image scale per unet encoder/decoder step')
parser.add_argument('--is_batchnorm', type=bool, default=True, help='use batchnorm instead of instancenorm')

# irrelevants
parser.add_argument('--val_bs', type=int, default=1, help='batch size at val time')
parser.add_argument('--val_step', type=int, default=200, help='do validation per val_step')
parser.add_argument('--draw_step', type=int, default=50, help='add train graphic result per draw_step')
parser.add_argument('--save_step', type=int, default=5000, help='save model and optimizer state dict per save_step')
parser.add_argument('--verbose', action='store_true', help='whether to display the loss information per iter')
parser.add_argument('--init', type=str, choices=['kaiming', 'xavier', 'normal', 'orthogonal'], default='kaiming', help='network weight init type')
parser.add_argument('--ckpt', type=str, choices=['best', 'last'], default='best', help='which kind of network is evaluated')

# baseline settings
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('-n', '--nl', action='store_true', help='whether to use negative learning')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')

args = parser.parse_args()
param = Parser(args).get_param()


def test(model):
    
    log = logging.getLogger()
    for hdlr in log.handlers[:]:
        log.removeHandler(hdlr)
    log.addHandler(logging.FileHandler(os.path.join(param.path.path_to_test, "test_log.txt"), mode='w'))
    log.addHandler(logging.StreamHandler(sys.stdout))
    
    save_model_path = os.path.join(param.path.path_to_model, f'{param.exp.exp_name}_{args.ckpt}_model.pth')
    state_dicts = torch.load(save_model_path, map_location='cpu')
    val_performance = state_dicts['metric']
    val_performance2 = 0
    if args.model == 'cps':
        val_performance2 = state_dicts['metric2']
    if val_performance2 > val_performance:
        model.load_state_dict(state_dicts['model_state_dict2'])
    else:
        model.load_state_dict(state_dicts['model_state_dict'])
        
    logging.info(f"init weight from {save_model_path},\
        performance on validation set is [{args.eval}] {max(val_performance, val_performance2)}")
    db_test = param.get_dataset(split='test')
    testloader = DataLoader(db_test, num_workers=1, batch_size=1)
    
    model.eval()
    test_all_case(model, param, testloader, stride_xy=64, stride_z=64, gpu_id=args.gpu)
 
    
def maybe_restore_model(*models, num_optim_models=1):
    optimizers = tuple(optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001) for model in models)[:num_optim_models]
    if args.restore:
        
        save_model_path = os.path.join(param.path.path_to_model, f'{param.exp.exp_name}_{args.ckpt}_model.pth')
        if not os.path.exists(save_model_path):
            print(f'the designated model path {save_model_path} does not exist')
            logging.info(msg=param)
            return models, optimizers
        
        state_dicts = torch.load(save_model_path, map_location='cpu')
        models[0].load_state_dict(state_dicts['model_state_dict'])
        optimizers[0].load_state_dict(state_dicts['optimizer_state_dict'])
        if num_optim_models == 2:
            models[1].load_state_dict(state_dicts['model2_state_dict'])
            optimizers[1].load_state_dict(state_dicts['optimizer2_state_dict'])
        else:
            RuntimeError(f'not configured for more than 2 models')
        base_lr = optimizers[0].param_groups[0]['lr']
        max_iter = param.exp.max_iter - state_dicts['iterations']
        
        assert max_iter > 0, f"restoring from a model trained more than current configured max_iteration {param.exp.max_iter}"
        logging.info(f"restoring from {save_model_path}, base_lr {param.exp.base_lr} -> {base_lr}, max_iter {param.exp.max_iter} -> {max_iter}")
        param.exp.base_lr = base_lr
        param.exp.max_iter = max_iter
    
    logging.info(msg=param)
    return models, optimizers


def main():
    global param
    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().addHandler(logging.FileHandler(os.path.join(param.path.path_to_snapshot, "log.txt"), mode='w'))
    model = None

    if args.model == 'unet':
        model = UNet(param).cuda(args.gpu)
        init_weights(model, args.init)
        train_unet(*maybe_restore_model(model), param, args)
    
    elif args.model == 'branched':
        if param.dataset.n_coarse > 2:
            model = UNetMultiBranchNetwork(param).cuda(args.gpu)
        elif param.dataset.n_coarse == 2:
            model = UNetSingleBranchNetwork(param).cuda(args.gpu)
        init_weights(model, args.init)
        train_c2f(*maybe_restore_model(model), param, args)
        
    elif args.model == 'cps':
        param.exp.pseudo_label = False
        param.exp.mixup_label = False
        param.exp.separate_norm = False
        param.exp.priority_cat = False
        if param.dataset.n_coarse > 2:
            model = UNetMultiBranchNetwork(param).cuda(args.gpu)
            model2 = UNetMultiBranchNetwork(param).cuda(args.gpu)
        elif param.dataset.n_coarse == 2:
            model = UNetSingleBranchNetwork(param).cuda(args.gpu)
            model2 = UNetSingleBranchNetwork(param).cuda(args.gpu)
        init_weights(model, 'kaiming')
        init_weights(model2, 'xavier')
        train_cps(*maybe_restore_model(model, model2, num_optim_models=2), param, args)
        
    elif args.model == 'uamt':
        param.exp.pseudo_label = False
        param.exp.mixup_label = False
        param.exp.separate_norm = False
        param.exp.priority_cat = False
        if param.dataset.n_coarse > 2:
            model = UNetMultiBranchNetwork(param).cuda(args.gpu)
            ema_model = UNetMultiBranchNetwork(param).cuda(args.gpu)
        elif param.dataset.n_coarse == 2:
            model = UNetSingleBranchNetwork(param).cuda(args.gpu)
            ema_model = UNetSingleBranchNetwork(param).cuda(args.gpu)
        init_weights(model, args.init)
        init_weights(ema_model, args.init)
        for params in ema_model.parameters():
            params.detach_()
        train_uamt(*maybe_restore_model(model, ema_model), param, args)
        
    test(model)
    print(f'train-test over for {param.exp.exp_name}')
    

if __name__ == '__main__':
    main()
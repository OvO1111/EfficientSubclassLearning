import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.utils import *
from collections import OrderedDict
from utils.parser import OmegaParser


class ForegroundBranch(nn.Module):
    def __init__(self, in_size, repeat, index, param: OmegaParser):
        super(ForegroundBranch).__init__()
        self.foreground_transformer = nn.Sequential(OrderedDict([
            ('trans', BatchNorm(in_size)),
            ('actv', nn.ReLU()),
            ('ext', ConvBlock(repeat=repeat,
                              in_channels=in_size, out_channels=in_size,
                              kernel_size=(3,)*math.floor(param.n_dim), padding=(1,)*math.floor(param.dataset.n_dim)))
        ]))
        self.priority_cat = param.p
        self.coarse_classifier = Conv(in_size, 1, 1)
        self.fine_classifier = Conv(in_size + param.p, len(param.mapping[index]), 1)
        
    def forward(self, inputs):
        out = self.foreground_transformer(inputs)
        coarse_logit = self.coarse_classifier(out)
        if self.priority_cat:
            fine_logit = self.fine_classifier(torch.cat([coarse_logit, out], dim=1))
        else:
            fine_logit = self.fine_classifier(out)
        return coarse_logit, fine_logit
    

class UNetMultiBranchClassifier(nn.Module):
    def __init__(self, in_size, repeat=1, param: OmegaParser=None):
        super(UNetMultiBranchClassifier, self).__init__()
        self.foreground_branches = nn.ModuleDict()
    
        if param.s:
            for c in range(1, param.n_coarse):
                self.foreground_branches[c] = ForegroundBranch(in_size, repeat, c)

            self.background_branch = nn.Sequential(OrderedDict([
                ('trans', BatchNorm(in_size)),
                ('actv', nn.ReLU()),
                ('ext', ConvBlock(repeat=repeat,
                                in_channels=in_size, out_channels=in_size,
                                kernel_size=(3,)*math.floor(param.n_dim), padding=(1,)*math.floor(param.n_dim))),
                ('classifier', Conv(in_size, 1, 1))
            ]))
        
        else:
            self.conv = nn.Sequential(OrderedDict([
                ('trans', BatchNorm(in_size)),
                ('actv', nn.ReLU()),
                ('ext', ConvBlock(repeat=repeat,
                                in_channels=in_size, out_channels=in_size,
                                kernel_size=(3,)*math.floor(param.n_dim), padding=(1,)*math.floor(param.n_dim)))
            ]))
            
            self.param = param
            self.coarse = Conv(in_size, param.n_coarse, 1)
            if param.p: self.fine = Conv(in_size + param.n_coarse, param.n_fine, 1)
            else: self.fine = Conv(in_size, param.n_fine, 1)
        
    def forward(self, inputs):

        if self.param.s:
            all_coarse_logits = torch.zeros((inputs.size(0), self.param.n_coarse) + inputs.shape[2:],
                                            device=inputs.device, dtype=torch.float32)
            all_fine_logits = torch.zeros((inputs.size(0), self.param.n_fine) + inputs.shape[2:],
                                          device=inputs.device, dtype=torch.float32)
            
            for c in range(1, self.param.n_coarse):
                coarse_logit, fine_logit = self.foreground_branches[c](inputs)
                all_coarse_logits[:, c] = coarse_logit
                all_fine_logits[:, slice(*self.param.mapping[c])] = fine_logit
                
            bg_logit = self.background_branch(inputs)
            all_coarse_logits[:, 0] = all_fine_logits[:, 0] = bg_logit
        
        else:
            inputs = self.conv(inputs)
            all_coarse_logits = self.coarse(inputs)
            if self.param.p: all_fine_logits = self.fine(torch.cat([all_coarse_logits, inputs]), dim=1)
            else: all_fine_logits = self.fine(inputs)
                
        return {'coarse_logit': all_coarse_logits, 'fine_logit': all_fine_logits}
        

class UNetMultiBranchNetwork(nn.Module):

    def __init__(self, param: OmegaParser):
        super(UNetMultiBranchNetwork, self).__init__()

        filters = [16 * 2 ** x for x in range(5)]

        # encoder
        self.encoder = UNetEncoder(
            in_chns=param.n_channels,
            feature_num=filters,
            layer_num=5
        )
        # decoder
        self.decoder = UNetDecoder(
            feature_num=filters,
            layer_num=5,
            output_feature_map=True,
            trunc_at_feature_map=True
        )
        # coarse to fine classification head
        self.param = param
        self.drop = nn.Dropout(0.3)
        self.classifier = UNetMultiBranchClassifier(filters[0], repeat=1, param=param)

    def forward(self, inputs):
        embed, skip = self.encoder(inputs)
        feat_map = self.decoder(embed, skip)
        drop_feat_map = self.drop(feat_map)
        outdict = self.classifier(drop_feat_map)
        outdict.update({'feature_map': feat_map})
        return outdict
    
    @torch.no_grad()
    def gen_pseudo_labels(self, q_im, q_soft, q_lc, threshold=0.4):
        rot_angle = random.randint(0, 3)
        flip_axis = random.randint(2, 3)
        
        k_im = torch.flip(torch.rot90(q_im, k=rot_angle, dims=(2, 3)), dims=(flip_axis,))
        
        k_out = self.forward(k_im)
        k_soft = torch.softmax(k_out['fine_logit'], dim=1)
        k_soft = torch.rot90(torch.flip(k_soft, dims=(flip_axis,)), k=-rot_angle, dims=(2, 3))
        
        # one-hot label
        pseudo_label = torch.zeros(q_soft.shape, dtype=torch.float32, device=q_soft.device)
        for i_label in range(self.param.n_fine):
            i_ = 0 if i_label == 0 else 1
            pseudo_label[:, i_label] = (k_soft[:, i_label] > threshold) & (q_soft[:, i_label] > threshold) & (q_lc == i_)

        return pseudo_label
    
    @torch.no_grad()
    def gen_mixup_labels(self, q_im, q_lc, q_soft, mixed_im, mixed_lf, alpha, threshold=0.4, with_pseudo_label=True):
        
        if math.floor(self.param.n_dim) == 3:
            mixed_lf = F.one_hot(mixed_lf, self.param.n_fine).permute(0, 4, 1, 2, 3)
        elif math.floor(self.param.n_dim) == 2:
            mixed_lf = F.one_hot(mixed_lf, self.param.n_fine).permute(0, 3, 1, 2)
        
        mixed_label = torch.zeros(mixed_lf.size(), device=mixed_lf.device, dtype=torch.float32)
        
        if with_pseudo_label:
            pseudo_label = self.gen_pseudo_labels(q_im, q_soft, q_lc, threshold)
            for i_batch in range(pseudo_label.size(0)):
                mixed_label[i_batch] = pseudo_label[i_batch] * (1 - alpha[i_batch]) + mixed_lf[i_batch] * alpha[i_batch]
        else:
            for i_batch in range(q_im.size(0)):
                mixed_label[i_batch] = mixed_lf[i_batch] * alpha[i_batch]
            
        mixed_pred = self.forward(mixed_im)['fine_logit']
        return mixed_pred, mixed_label
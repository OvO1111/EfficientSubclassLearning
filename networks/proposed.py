
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from meta import db as param
from collections import defaultdict, OrderedDict
from networks.utils import init_weights, ConvNd, BatchNormNd, MaxPoolNd


class UnetConv(nn.Module):
    def __init__(self, in_size, out_size, is_separate_batchnorm, kernel_size=None, padding_size=None, init_stride=None):
        super(UnetConv, self).__init__()
        
        if kernel_size is None:
            kernel_size = (3,) * param.dataset.n_dim
            padding_size = (1,) * param.dataset.n_dim
            init_stride = 1

        if is_separate_batchnorm:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', ConvNd(in_size, out_size, kernel_size, init_stride, padding_size)),
                ('bn', BatchNormNd(out_size)),
                ('nl', nn.ReLU(inplace=True)),
            ]))
            if param.exp.separate_norm:
                self.conv2 = ConvNd(out_size, out_size, kernel_size, init_stride, padding_size)
            else:
                self.conv2 = nn.Sequential(OrderedDict([
                    ('conv', ConvNd(out_size, out_size, kernel_size, init_stride, padding_size)),
                    ('bn', BatchNormNd(out_size)),
                    ('nl', nn.ReLU(inplace=True)),
                ]))
        else:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', ConvNd(in_size, out_size, kernel_size, init_stride, padding_size)),
                ('nl', nn.ReLU(inplace=True)),
            ]))
            self.conv2 = nn.Sequential(OrderedDict([
                ('conv', ConvNd(out_size, out_size, kernel_size, init_stride, padding_size)),
                ('nl', nn.ReLU(inplace=True)),
            ]))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUpConcat(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUpConcat, self).__init__()
        self.conv = UnetConv(in_size + out_size, out_size, is_batchnorm)
        if param.dataset.n_dim == 3:
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        elif param.dataset.n_dim == 2:
            self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        else:
            self.up = None

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        out1 = self.conv(torch.cat([outputs1, outputs2], 1))
        
        return out1


class ConvBlock(nn.Module):
    def __init__(self, repeat=2, *args, **kwargs):
        super(ConvBlock, self).__init__()
        conv_block = [ConvNd(*args, **kwargs) for _ in range(repeat)]
        relu_block = [nn.ReLU(inplace=True) for _ in range(repeat)]
        conv = [None] * (2 * repeat)
        conv[::2] = conv_block
        conv[1::2] = relu_block
        self.conv = nn.Sequential(*conv)

    def forward(self, inputs):
        feat = self.conv(inputs)
        return feat
    

class UnetC2FOutput(nn.Module):
    def __init__(self, in_size, repeat=1):
        super(UnetC2FOutput, self).__init__()

        if param.exp.separate_norm:
            self.coarse_foreground = nn.Sequential(
                BatchNormNd(in_size),
                nn.ReLU(),
                ConvBlock(
                    repeat=repeat, in_channels=in_size,
                    out_channels=in_size, kernel_size=(3,) * param.dataset.n_dim, padding=(1,) * param.dataset.n_dim
                ),
            )
            self.coarse_feat2logit = ConvNd(in_size, 1, 1)
            if param.exp.priority_cat:
                self.coarse_feat2feat = ConvNd(in_size + 1, param.dataset.n_fine - 1, 1)
            else:
                self.coarse_feat2feat = ConvNd(in_size, param.dataset.n_fine - 1, 1)
            self.coarse_background = nn.Sequential(
                BatchNormNd(in_size),
                nn.ReLU(),
                ConvBlock(
                    repeat=repeat, in_channels=in_size,
                    out_channels=in_size, kernel_size=(3,) * param.dataset.n_dim, padding=(1,) * param.dataset.n_dim
                ),
                ConvNd(in_size, 1, 1),
            )
        
        elif param.exp.priority_cat:
            self.conv = nn.Sequential(
                BatchNormNd(in_size),
                nn.ReLU(inplace=True),
                ConvBlock(
                    repeat=repeat, in_channels=in_size,
                    out_channels=in_size, kernel_size=(3,) * param.dataset.n_dim, padding=(1,) * param.dataset.n_dim
                ),
            )
            self.coarse = ConvNd(in_size, param.dataset.n_coarse, 1)
            self.fine = ConvNd(in_size + param.dataset.n_coarse, param.dataset.n_fine, 1)

        else:
            self.conv = nn.Sequential(
                BatchNormNd(in_size),
                nn.ReLU(inplace=True),
                ConvBlock(
                    repeat=repeat, in_channels=in_size,
                    out_channels=in_size, kernel_size=(3,) * param.dataset.n_dim, padding=(1,) * param.dataset.n_dim
                ),
            )
            self.coarse = ConvNd(in_size, param.dataset.n_coarse, 1)
            self.fine = ConvNd(in_size, param.dataset.n_fine, 1)
        
    def forward(self, inputs):

        if param.exp.separate_norm:
            foreground = self.coarse_foreground(inputs)
            
            fg_logit = self.coarse_feat2logit(foreground)
            bg_logit = self.coarse_background(inputs)
            
            fg_concat = torch.cat([foreground, fg_logit], dim=1)
            if param.exp.priority_cat:
                fine_split = self.coarse_feat2feat(fg_concat)
            else:
                fine_split = self.coarse_feat2feat(foreground)
            
            coarse = torch.cat([bg_logit, fg_logit], dim=1)
            fine = torch.cat([bg_logit, fine_split], dim=1)
            return {'coarse_logit': coarse, 'fine_logit': fine}
        
        elif param.exp.priority_cat:
            inputs = self.conv(inputs)
            coarse = self.coarse(inputs)
            fine = torch.cat([inputs, coarse], dim=1)
            fine = self.fine(fine)
            return {'coarse_logit': coarse, 'fine_logit': fine}
        
        else:
            inputs = self.conv(inputs)
            coarse = self.coarse(inputs)
            fine = self.fine(inputs)
            return {'coarse_logit': coarse, 'fine_logit': fine}
        

class CU3D(nn.Module):

    def __init__(self, **kwargs):
        super(CU3D, self).__init__()
        self.in_channels = kwargs.get('in_channels', 1)
        self.is_batchnorm = kwargs.get('is_batchnorm', True)
        self.feature_scale = kwargs.get('feature_scale', 4)

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = MaxPoolNd(kernel_size=(2,) * param.dataset.n_dim)
        
        self.conv2 = UnetConv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = MaxPoolNd(kernel_size=(2,) * param.dataset.n_dim)
        
        self.conv3 = UnetConv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = MaxPoolNd(kernel_size=(2,) * param.dataset.n_dim)
        
        self.conv4 = UnetConv(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = MaxPoolNd(kernel_size=(2,) * param.dataset.n_dim)

        self.center = UnetConv(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUpConcat(filters[4], filters[3], self.is_batchnorm)
        self.up_concat3 = UnetUpConcat(filters[3], filters[2], self.is_batchnorm)
        self.up_concat2 = UnetUpConcat(filters[2], filters[1], self.is_batchnorm)
        self.up_concat1 = UnetUpConcat(filters[1], filters[0], self.is_batchnorm)

        # final conv (without any concat)
        self.final = UnetC2FOutput(filters[0], repeat=1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialize weights
        for m in self.modules():
            if isinstance(m, ConvNd):
                init_weights(m.conv, init_type='kaiming')
            elif isinstance(m, BatchNormNd):
                init_weights(m.norm, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
       
        center = self.center(maxpool4)
        center = self.dropout1(center)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        
        outdict = self.final(self.dropout2(up1))
        outdict.update({'feature_map': up1})
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
        for i_label in range(param.dataset.n_fine):
            i_ = 0 if i_label == 0 else 1
            pseudo_label[:, i_label] = (k_soft[:, i_label] > threshold) & (q_soft[:, i_label] > threshold) & (q_lc == i_)

        return pseudo_label
    
    @torch.no_grad()
    def gen_mixup_labels(self, q_im, q_lc, q_soft, mixed_im, mixed_lf, alpha, threshold=0.4, with_pseudo_label=True):
        
        if param.dataset.n_dim == 3:
            mixed_lf = F.one_hot(mixed_lf, param.dataset.n_fine).permute(0, 4, 1, 2, 3)
        elif param.dataset.n_dim == 2:
            mixed_lf = F.one_hot(mixed_lf, param.dataset.n_fine).permute(0, 3, 1, 2)
        
        mixed_label = torch.zeros(mixed_lf.size(), device=mixed_lf.device, dtype=torch.float32)
        
        if with_pseudo_label:
            pseudo_label = self.gen_pseudo_labels(q_im, q_soft, q_lc, threshold)
            for i_batch in range(pseudo_label.size(0)):
                mixed_label[i_batch] = pseudo_label[i_batch] * (1 - alpha[i_batch]) + mixed_lf[i_batch] * alpha[i_batch]
        else:
            for i_batch in range(pseudo_label.size(0)):
                mixed_label[i_batch] = mixed_lf[i_batch] * alpha[i_batch]
            
        mixed_pred = self.forward(mixed_im)['fine_logit']
        return mixed_pred, mixed_label
    

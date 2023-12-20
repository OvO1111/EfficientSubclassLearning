
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict, OrderedDict
from networks.utils import init_weights, set_ndim, Conv, BatchNorm, MaxPool


param = None


class UnetConv(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=None, padding_size=None, init_stride=None):
        super(UnetConv, self).__init__()
        
        if kernel_size is None:
            kernel_size = (3,) * math.floor(param.dataset.n_dim)
            padding_size = (1,) * math.floor(param.dataset.n_dim)
            init_stride = 1

        if is_batchnorm:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', Conv(in_size, out_size, kernel_size, init_stride, padding_size)),
                ('bn', BatchNorm(out_size)),
                ('nl', nn.ReLU(inplace=True)),
            ]))
            if param.exp.separate_norm:
                self.conv2 = Conv(out_size, out_size, kernel_size, init_stride, padding_size)
            else:
                self.conv2 = nn.Sequential(OrderedDict([
                    ('conv', Conv(out_size, out_size, kernel_size, init_stride, padding_size)),
                    ('bn', BatchNorm(out_size)),
                    ('nl', nn.ReLU(inplace=True)),
                ]))
        else:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', Conv(in_size, out_size, kernel_size, init_stride, padding_size)),
                ('nl', nn.ReLU(inplace=True)),
            ]))
            self.conv2 = nn.Sequential(OrderedDict([
                ('conv', Conv(out_size, out_size, kernel_size, init_stride, padding_size)),
                ('nl', nn.ReLU(inplace=True)),
            ]))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUpConcat(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UnetUpConcat, self).__init__()
        self.conv = UnetConv(in_size + out_size, out_size, is_batchnorm)
        if math.floor(param.dataset.n_dim) == 3:
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        elif math.floor(param.dataset.n_dim) == 2:
            self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        else:
            self.up = None

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        out1 = self.conv(torch.cat([outputs1, outputs2], 1))
        
        return out1
        

class UNet(nn.Module):

    def __init__(self, parameter):
        super(UNet, self).__init__()
        global param
        param = parameter
        set_param(parameter)
        self.in_channels = param.dataset.n_mode
        self.is_batchnorm = param.network.is_batchnorm
        self.feature_scale = param.network.feature_scale

        filters = [param.network.base_feature_num * self.feature_scale ** x for x in range(5)]

        # downsampling
        self.conv1 = UnetConv(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = MaxPool(kernel_size=(2,) * math.floor(param.dataset.n_dim))
        
        self.conv2 = UnetConv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = MaxPool(kernel_size=(2,) * math.floor(param.dataset.n_dim))
        
        self.conv3 = UnetConv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = MaxPool(kernel_size=(2,) * math.floor(param.dataset.n_dim))
        
        self.conv4 = UnetConv(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = MaxPool(kernel_size=(2,) * math.floor(param.dataset.n_dim))

        self.center = UnetConv(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUpConcat(filters[4], filters[3], self.is_batchnorm)
        self.up_concat3 = UnetUpConcat(filters[3], filters[2], self.is_batchnorm)
        self.up_concat2 = UnetUpConcat(filters[2], filters[1], self.is_batchnorm)
        self.up_concat1 = UnetUpConcat(filters[1], filters[0], self.is_batchnorm)

        # final conv (without any concat)
        self.final = Conv(filters[0], param.dataset.n_fine, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

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
        
        outdict = dict(logit=self.final(self.dropout2(up1)), feature_map=up1)
        return outdict


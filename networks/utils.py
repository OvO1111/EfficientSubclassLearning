import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from collections import OrderedDict

param = None

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(f'initialization method {init_type} is not implemented')
    
    
class ConvNd(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConvNd, self).__init__()
        self.conv = None
        if param.dataset.n_dim == 2:
            self.conv = nn.Conv2d(*args, **kwargs)
        elif param.dataset.n_dim == 3:
            self.conv = nn.Conv3d(*args, **kwargs)
        for m in self.children():
            init_weights(m, init_type='kaiming')
    
    def forward(self, inputs):
        return self.conv(inputs)


class MaxPoolNd(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MaxPoolNd, self).__init__()
        self.maxpool = None
        if param.dataset.n_dim == 2:
            self.maxpool = nn.MaxPool2d(*args, **kwargs)
        elif param.dataset.n_dim == 3:
            self.maxpool = nn.MaxPool3d(*args, **kwargs)
    
    def forward(self, inputs):
        return self.maxpool(inputs)


class BatchNormNd(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BatchNormNd, self).__init__()
        self.norm = None
        if param.dataset.n_dim == 2:
            self.norm = nn.BatchNorm2d(*args, **kwargs)
        elif param.dataset.n_dim == 3:
            self.norm = nn.BatchNorm3d(*args, **kwargs)
    
    def forward(self, inputs):
        return self.norm(inputs)
    
    
class ConvBlock(nn.Module):
    def __init__(self, repeat=2, **kwargs):
        super(ConvBlock, self).__init__()
        self.repeat = repeat
        self.conv = nn.ModuleList()
        for _ in self.repeat:
            self.conv.append(nn.Sequential(OrderedDict([
                ('conv', ConvNd(**kwargs)),
                ('actv', nn.LeakyReLU())
            ])))
        
    def forward(self, inputs):
        for rep in self.repeat:
            inputs = self.conv[rep](inputs)
        return inputs
    
    
class UNetEncoderStep(nn.Module):
    def __init__(self, in_chns, out_chns, kernel_size, stride_size, padding_size, ds=True):
        super(UNetEncoderStep, self).__init__()
        self.convs = nn.Sequential(OrderedDict([
            ('conv1', ConvNd(in_chns, out_chns, kernel_size, stride_size, padding_size)),
            ('norm1', BatchNormNd(out_chns)),
            ('actv1', nn.LeakyReLU()),
            ('conv2', ConvNd(out_chns, out_chns, kernel_size, stride_size, padding_size)),
            ('norm2', BatchNormNd(out_chns)),
            ('actv2', nn.LeakyReLU()),
        ]))
        self.down = MaxPoolNd(kernel_size=(2, 2, 2))
        self.down_sampling = ds
        
        for m in self.children():
            if m.__class__.__name__.find('Conv') != -1: continue
            init_weights(m, init_type='kaiming')
        
    def forward(self, inputs):
        out = self.convs(inputs)
        if self.down_sampling:
            out = self.down(out)
        return out


class UnetDecoderStep(nn.Module):
    def __init__(self, in_chns, out_chns, kernel_size, stride_size, padding_size):
        super(UnetDecoderStep, self).__init__()
        self.up = nn.Upsample(
            scale_factor=(param.network.image_scale,) * param.dataset.n_dim,
            mode='trilinear' if param.dataset.n_dim == 3 else 'bilinear', align_corners=False)
        self.convs = nn.Sequential(OrderedDict([
            ('conv1', ConvNd(in_chns + out_chns, out_chns, kernel_size, stride_size, padding_size)),
            ('norm1', BatchNormNd(out_chns)),
            ('actv1', nn.LeakyReLU()),
            ('conv2', ConvNd(out_chns, out_chns, kernel_size, stride_size, padding_size)),
            ('norm2', BatchNormNd(out_chns)),
            ('actv2', nn.LeakyReLU()),
        ]))
        
        for m in self.children():
            if m.__class__.__name__.find('Conv') != -1: continue
            init_weights(m, init_type='kaiming')
        
    def forward(self, inputs, skip):
        inputs = self.up(inputs)
        offset = inputs.size()[2] - skip.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        skip = f.pad(skip, padding)
        out = torch.cat([skip, inputs], dim=1)
        out = self.convs(out)
        return out
    
    
class UNetEncoder(nn.Module):
    default_params = dict(
        in_chns=3,
        kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1,
        feature_num=(16, 32, 64, 128, 256), layer_num=5
    )
    
    def __init__(self, **kwargs):
        super(UNetEncoder, self).__init__()
        self.encoder_steps = nn.ModuleDict()
        self.stride = kwargs.get('stride', self.default_params['stride'])
        self.in_chns = kwargs.get('in_chns', self.default_params['in_chns'])
        self.padding = kwargs.get('padding', self.default_params['padding'])
        self.layer_num = kwargs.get('layer_num', self.default_params['layer_num'])
        self.kernel_size = kwargs.get('kernel_size', self.default_params['kernel_size'])
        self.feature_num = kwargs.get('feature_num', self.default_params['feature_num'])
        
        assert len(self.feature_num) == self.layer_num
        self.feature_num = [self.in_chns] + self.feature_num
         
        for layer in range(self.layer_num - 1):
            self.encoder_steps[layer] = UNetEncoderStep(
                self.feature_num[layer],
                self.feature_num[layer+1], 
                self.kernel_size, self.padding
            )
        self.center = ConvNd(self.feature_num[-2], self.feature_num[-1], self.kernel_size, self.stride, self.padding)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, inputs):
        for layer in range(self.layer_num):
            inputs = self.encoder_steps[layer](inputs)
        out = self.center(inputs)
        out = self.dropout(out)
        return out
    
    
class UNetDecoder(nn.Module):
    default_params = dict(
        out_chns=4,
        kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1,
        feature_num=(16, 32, 64, 128, 256), layer_num=5, 
        output_feature_map = True,
        trunc_at_feature_map = False
    )
    
    def __init__(self, **kwargs):
        super(UNetDecoder, self).__init__()
        self.decoder_steps = nn.ModuleDict()
        self.stride = kwargs.get('stride', self.default_params['stride'])
        self.out_chns = kwargs.get('out_chns', self.default_params['out_chns'])
        self.padding = kwargs.get('padding', self.default_params['padding'])
        self.layer_num = kwargs.get('layer_num', self.default_params['layer_num'])
        self.kernel_size = kwargs.get('kernel_size', self.default_params['kernel_size'])
        self.feature_num = kwargs.get('feature_num', self.default_params['feature_num'])
        self.output_feature_map = kwargs.get('output_feature_map', self.default_params['output_feature_map'])
        self.trunc_at_feature_map = kwargs.get('trunc_at_feature_map', self.default_params['trunc_at_feature_map'])
        
        assert len(self.feature_num) == self.layer_num
        self.feature_num = self.feature_num[::-1] + [self.out_chns]
         
        for layer in range(self.layer_num-1):
            self.decoder_steps[layer] = UnetDecoderStep(
                self.feature_num[layer],
                self.feature_num[layer+1], 
                self.kernel_size, self.padding
            )
        if not self.trunc_at_feature_map:
            self.classifier = nn.Conv3d(self.feature_num[-2], self.feature_num[-1], 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, inputs):
        for layer in range(self.layer_num):
            inputs = self.decoder_steps[layer](inputs)
        out = self.dropout(inputs)
        if self.trunc_at_feature_map:
            return inputs

        out = self.classifier(out)
        if self.output_feature_map:
            return out, inputs
        return out


def set_param(parameter):
    global param
    param = parameter
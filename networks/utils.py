import torch.nn as nn
from torch.nn import init

param = None

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
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
    
    
def set_param(parameter):
    global param
    param = parameter
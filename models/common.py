import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from .downsampler import Downsampler
from utils.common_utils import np_to_torch

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)

class AddNoisyFMs(nn.Module):
    def __init__(self, dim, sigma=1):
        super(AddNoisyFMs, self).__init__()
        self.dim = dim
        self.sigma = sigma

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim

        b = torch.zeros(a, dtype=input.dtype)#.type_as(input.data)
        b.normal_(std=self.sigma)

        return torch.cat((input, b), axis=1)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        return b

        # x = torch.autograd.Variable(b)
        #
        # return x

class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        # self.alpha = torch.Tensor([alpha])
        self.p = p

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            # if self.alpha.device != x.device:
            #     self.alpha = self.alpha.to(x.device)
            epsilon = torch.randn(x.size(), device=x.device, dtype=x.dtype)  * (self.p / (1 - self.p)) + 1 # * self.alpha + 1

            # epsilon = Variable(epsilon)
            # if x.is_cuda:
            #     epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x

class GaussianDropout2d(nn.Module):
    def __init__(self, p, layer):
        super(GaussianDropout2d, self).__init__()
        self.p = p
        self.layer = layer

    def forward(self, x):
        mu = self.layer(x)
        std = F.conv2d(x**2, self.layer.weight.data**2, stride=self.layer.stride, padding=self.layer.padding)
        std = (self.p / (1 - self.p) * std).sqrt()
        eps = std.data.new(std.size()).normal_()
        return mu + std * eps

class VariationalDropout2d(nn.Module):
    def __init__(self, p, layer):
        super(VariationalDropout2d, self).__init__()
        self.p = p
        self.layer = layer
        log_alpha = (self.p / (1 - self.p) * torch.ones(self.layer.out_channels)).log()
        self.log_alpha = nn.Parameter(log_alpha)


    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921

        alpha = self.log_alpha.exp()

        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha**2 + c3 * alpha**3

        kl = -negative_kl

        return kl.mean()

    def forward(self, x):
        mu = self.layer(x)
        std = F.conv2d(x**2, self.layer.weight.data**2, stride=self.layer.stride)
        alpha = self.log_alpha.exp()
        alpha = torch.clamp(alpha, min=alpha.min().item(), max=1)
        std = (alpha[None, :, None, None] * std).sqrt()
        eps = std.data.new(std.size()).normal_()
        return mu + std * eps

class VariationalDropout(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()

        self.dim = dim
        self.max_alpha = alpha
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921

        alpha = self.log_alpha.exp()

        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha**2 + c3 * alpha**3

        kl = -negative_kl

        return kl.mean()

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = Variable(torch.randn(x.size()))
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x

class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)

def conv(in_f, out_f, kernel_size, stride=1, ffg=False, implicit=False, initial_log_alpha=-5.0, log_alpha_grad=True, bias=True, pad='zero', downsample_mode='stride', dropout_mode=None, dropout_p=0.2, name='FFGConv'):

    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    # if not ffg:# and dropout_mode not in ['gaussian2d', 'variational2d']:
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    # elif ffg and not implicit:
    #     convolver = FFGConv(in_f, out_f, kernel_size, padding=to_pad, stride=stride, bias=bias, name=name)
    # elif ffg and implicit:
    #     convolver= FFGConv_implicit(in_f, out_f, kernel_size, padding=to_pad, stride=stride, bias=bias, name=name)

    dropout = None
    if dropout_mode == '2d' and not ffg:
        dropout = nn.Dropout2d(p=dropout_p)
    elif dropout_mode == '1d' and not ffg:
        dropout = nn.Dropout(p=dropout_p)
    elif dropout_mode == 'gaussian2d' and not ffg:
        dropout = GaussianDropout2d(p=dropout_p, layer=convolver)
        convolver = None
    elif dropout_mode == 'variational2d' and not ffg:
        dropout = VariationalDropout2d(p=dropout_p, layer=convolver)
        convolver = None
    elif dropout_mode == 'prob2d' and not ffg:
        probs = dropout_p * np.ones(out_f)
        dropout = ProbabilityDropout2d(probs=probs)
    elif dropout_mode == 'prob1d' and not ffg:
        probs = dropout_p * np.ones(out_f)
        dropout = ProbabilityDropout(probs=probs)
    elif dropout_mode == 'gaussian' and not ffg:
        dropout = GaussianDropout(alpha=dropout_p)
    elif dropout_mode == 'variational' and not ffg:
        dropout = VariationalDropout()

    layers = list(filter(lambda x: x is not None, [padder, convolver, dropout, downsampler]))

    if len(layers) == 1:
        return layers[0]
    else:
        return nn.Sequential(*layers)

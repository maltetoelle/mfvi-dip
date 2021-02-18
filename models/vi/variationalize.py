from typing import Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv2dRT, Conv2dLRT, Conv3dRT, Conv3dLRT
from .linear import LinearRT, LinearLRT
from .dropout import MCDropout

class MeanFieldVI(nn.Module):

    def __init__(self,
                 net: nn.Module,
                 prior: Dict[str, float] = None,
                 posteriors: Dict[str, float] = None,
                 beta: float = 1.,
                 kl_type: str = 'reverse',
                 reparam: str = 'local'):

        super(MeanFieldVI, self).__init__()
        self.net = net

        if reparam == 'local':
            self._conv3d = Conv3dLRT
            self._conv2d = Conv2dLRT
            self._linear = LinearLRT
        else:
            self._conv3d = Conv3dRT
            self._conv2d = Conv2dRT
            self._linear = LinearRT

        self._replace_deterministic_modules(self.net, prior, posteriors, kl_type)

        # self.net.kl = self.kl
        self.beta = torch.tensor([beta])

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @property
    def kl(self) -> Tensor:
        kl = 0
        for layer in self.modules():
            if hasattr(layer, '_kl'):
                kl += layer._kl
        return self.beta.to(kl.device) * kl

    def _replace_deterministic_modules(self,
                                       module: nn.Module,
                                       prior: Dict[str, float],
                                       posteriors: Dict[str, float],
                                       kl_type: str):

        for key, _module in module._modules.items():
            if len(_module._modules):
                self._replace_deterministic_modules(_module, prior, posteriors, kl_type)
            else:
                if isinstance(_module, nn.Linear):
                    layer = self._linear(
                        _module.in_features,
                        _module.out_features,
                        torch.is_tensor(_module.bias))
                    module._modules[key] = layer
                elif isinstance(_module, nn.Conv2d):
                    layer = self._conv2d(
                        in_channels=_module.in_channels,
                        out_channels=_module.out_channels,
                        kernel_size=_module.kernel_size,
                        bias=torch.is_tensor(_module.bias),
                        stride=_module.stride,
                        padding=_module.padding,
                        dilation=_module.dilation,
                        groups=_module.groups,
                        prior=prior,
                        posteriors=posteriors,
                        kl_type=kl_type)
                    module._modules[key] = layer
                elif isinstance(_module, nn.Conv3d):
                    layer = self._conv3d(
                        in_channels=_module.in_channels,
                        out_channels=_module.out_channels,
                        kernel_size=_module.kernel_size,
                        bias=torch.is_tensor(_module.bias),
                        stride=_module.stride,
                        padding=_module.padding,
                        dilation=_module.dilation,
                        groups=_module.groups,
                        prior=prior,
                        posteriors=posteriors,
                        kl_type=kl_type)
                    module._modules[key] = layer


class MCDropoutVI(nn.Module):

    def __init__(self,
                 net: nn.Module,
                 dropout_type: str = '1d',
                 dropout_p: float = 0.5,
                 deterministic_output: bool = False,
                 output_dip_drop: bool = False):

        super(MCDropoutVI, self).__init__()
        self.net = net
        self.dropout_type = dropout_type
        self.dropout_p = dropout_p

        self._replace_deterministic_modules(self.net)
        # self.deterministic_output = deterministic_output
        if deterministic_output:
            self._make_last_layer_deterministic(self.net)
        if not output_dip_drop:
            self._dip_make_output_deterministic(self.net)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def _replace_deterministic_modules(self, module: nn.Module):
        for key, _module in module._modules.items():
            if len(_module._modules):
                self._replace_deterministic_modules(_module)
            else:
                if isinstance(_module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                    module._modules[key] =  MCDropout(_module, self.dropout_type, self.dropout_p)

    def _make_last_layer_deterministic(self, module: nn.Module):
        for i, (key, layer) in enumerate(module._modules.items()):
            if i == len(module._modules) - 1:
                if isinstance(layer, MCDropout):
                    module._modules[key] = layer.layer
                elif len(layer._modules):
                    self._make_last_layer_deterministic(layer)

    def _dip_make_output_deterministic(self, module: nn.Module):
        for i, (key, layer) in enumerate(module._modules.items()):
            if type(layer) == nn.Sequential:
                for name, m in layer._modules.items():
                    if type(m) == MCDropout:
                        layer._modules[name] = m.layer

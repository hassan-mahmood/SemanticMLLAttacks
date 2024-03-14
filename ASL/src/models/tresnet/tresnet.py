# import sys
# import os
# sys.path.append('./')
# import torch
# import torch.nn as nn
# from torch.nn import Module as Module
# from collections import OrderedDict
# #from .layers.anti_aliasing import AntiAliasDownsampleLayer
# #from .layers.avg_pool import FastAvgPool2d
# #from .layers.general_layers import SEModule, SpaceToDepthModule

# import sys

# from .layers.anti_aliasing import AntiAliasDownsampleLayer
# from .layers.avg_pool import FastAvgPool2d
# # from src_files.ml_decoder.ml_decoder import MLDecoder
# from .layers.general_layers import SEModule, SpaceToDepthModule
# from inplace_abn import InPlaceABN, ABN
# import torch.nn.parallel
# import numpy as np
# import torch.nn.functional as F

# from ...mldecoder import *


# def InplacABN_to_ABN(module: nn.Module) -> nn.Module:
#     # convert all InplaceABN layer to bit-accurate ABN layers.
#     if isinstance(module, InPlaceABN):
#         module_new = ABN(module.num_features, activation=module.activation,
#                          activation_param=module.activation_param)
#         for key in module.state_dict():
#             module_new.state_dict()[key].copy_(module.state_dict()[key])
#         module_new.training = module.training
#         module_new.weight.data = module_new.weight.abs() + module_new.eps
#         return module_new
#     for name, child in reversed(module._modules.items()):
#         new_child = InplacABN_to_ABN(child)
#         if new_child != child:
#             module._modules[name] = new_child
#     return module

# def conv2d(ni, nf, stride):
#     return nn.Sequential(
#         nn.Conv2d(ni, nf, kernel_size=3, stride=stride, padding=1, bias=False),
#         nn.BatchNorm2d(nf),
#         nn.ReLU(inplace=True)
#     )


# def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
#     return nn.Sequential(
#         nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
#                   bias=False),
#         InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param)
#     )


# class BasicBlock(Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
#         super(BasicBlock, self).__init__()
#         if stride == 1:
#             self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
#         else:
#             if anti_alias_layer is None:
#                 self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
#             else:
#                 self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
#                                            anti_alias_layer(channels=planes, filt_size=3, stride=2))

#         self.conv2 = conv2d_ABN(planes, planes, stride=1, activation="identity")
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         reduce_layer_planes = max(planes * self.expansion // 4, 64)
#         self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

#     def forward(self, x):
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         else:
#             residual = x

#         out = self.conv1(x)
#         out = self.conv2(out)

#         if self.se is not None: out = self.se(out)

#         out += residual

#         out = self.relu(out)

#         return out


# class Bottleneck(Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
#                                 activation_param=1e-3)
#         if stride == 1:
#             self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
#                                     activation_param=1e-3)
#         else:
#             if anti_alias_layer is None:
#                 self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
#                                         activation_param=1e-3)
#             else:
#                 self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1,
#                                                       activation="leaky_relu", activation_param=1e-3),
#                                            anti_alias_layer(channels=planes, filt_size=3, stride=2))

#         self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
#                                 activation="identity")

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#         reduce_layer_planes = max(planes * self.expansion // 8, 64)
#         self.se = SEModule(planes, reduce_layer_planes) if use_se else None

#     def forward(self, x):
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         else:
#             residual = x

#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.se is not None: out = self.se(out)

#         out = self.conv3(out)
#         out = out + residual  # no inplace
#         out = self.relu(out)

#         return out


# class TResNet(Module):

#     def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, first_two_layers=BasicBlock):
#         super(TResNet, self).__init__()

#         # JIT layers
#         space_to_depth = SpaceToDepthModule()
#         anti_alias_layer = AntiAliasDownsampleLayer
#         global_pool_layer = FastAvgPool2d(flatten=True)

#         # TResnet stages
#         self.inplanes = int(64 * width_factor)
#         self.planes = int(64 * width_factor)
#         conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
#         layer1 = self._make_layer(first_two_layers, self.planes, layers[0], stride=1, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 56x56
#         layer2 = self._make_layer(first_two_layers, self.planes * 2, layers[1], stride=2, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 28x28
#         layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 14x14
#         layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
#                                   anti_alias_layer=anti_alias_layer)  # 7x7

#         # body
#         self.body = nn.Sequential(OrderedDict([
#             ('SpaceToDepth', space_to_depth),
#             ('conv1', conv1),
#             ('layer1', layer1),
#             ('layer2', layer2),
#             ('layer3', layer3),
#             ('layer4', layer4)]))

#         # head
#         self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
#         self.num_features = (self.planes * 8) * Bottleneck.expansion
#         fc = nn.Linear(self.num_features, num_classes)
#         self.head = nn.Sequential(OrderedDict([('fc', fc)]))

#         # model initilization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # residual connections special initialization
#         for m in self.modules():
#             if isinstance(m, BasicBlock):
#                 m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
#             if isinstance(m, Bottleneck):
#                 m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
#             if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

#     def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             layers = []
#             if stride == 2:
#                 # avg pooling before 1x1 conv
#                 layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
#             layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
#                                   activation="identity")]
#             downsample = nn.Sequential(*layers)

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
#                             anti_alias_layer=anti_alias_layer))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks): layers.append(
#             block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
#         return nn.Sequential(*layers)

#     def forward(self, x,par=None):
#         x = self.body(x)
#         self.embeddings = self.global_pool(x)
#         logits = self.head(self.embeddings)
#         return logits, None


# def TResnetS(model_params):
#     """Constructs a small TResnet model.
#     """
#     in_chans = 3
#     num_classes = model_params['num_classes']
#     args = model_params['args']
#     model = TResNet(layers=[3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans)
#     return model

# def TResnetM(model_params):
#     """Constructs a medium TResnet model.
#     """
#     in_chans = 3
#     num_classes = model_params['num_classes']
#     model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans)
#     return model

# def TResnetMLDecoder(model_params):
#     """Constructs a large TResnet model.
#     """
#     in_chans = 3
#     num_classes = model_params['num_classes']
#     layers_list = [3, 4, 23, 3]
#     model = TResNet(layers=layers_list, num_classes=num_classes, in_chans=in_chans, first_two_layers=Bottleneck)
#     return model

# def TResnetL(model_params):
#     """Constructs a large TResnet model.
#     """
#     in_chans = 3
#     num_classes = model_params['num_classes']
#     layers_list = [3, 4, 23, 3]
#     model = TResNet(layers=layers_list, num_classes=num_classes, in_chans=in_chans, first_two_layers=Bottleneck)
#     return model

# def TResnetXL(model_params):
#     """Constructs a large TResnet model.
#     """
#     in_chans = 3
#     num_classes = model_params['num_classes']
#     layers_list = [3, 8, 34, 5]
#     model = TResNet(layers=layers_list, num_classes=num_classes, in_chans=in_chans, first_two_layers=Bottleneck)
#     return model

###############################
# works for asl
import sys
import os
sys.path.append('./')
import torch
import torch.nn as nn
from torch.nn import Module as Module
from collections import OrderedDict
#from .layers.anti_aliasing import AntiAliasDownsampleLayer
#from .layers.avg_pool import FastAvgPool2d
#from .layers.general_layers import SEModule, SpaceToDepthModule
from inplace_abn import InPlaceABN
import torch.nn.parallel
import numpy as np
import torch.nn.functional as F

import sys
from ...mldecoder import *
# sys.path.append('ASL/')
# from src import ml_decoder



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class SpaceToDepthModule(nn.Module):
    def __init__(self, remove_model_jit=False):
        super().__init__()
        if not remove_model_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


@torch.jit.script
class SpaceToDepthJit(object):
    def __call__(self, x: torch.Tensor):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 16, H // 4, W // 4)  # (N, C*bs^2, H//bs, W//bs)
        return x


class hard_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hard_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.add_(3.).clamp_(0., 6.).div_(6.)
        else:
            return F.relu6(x + 3.) / 6.


class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(SEModule, self).__init__()
        self.avg_pool = FastAvgPool2d()
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        # self.activation = hard_sigmoid(inplace=inplace)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = self.relu(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se


class FastAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)




class AntiAliasDownsampleLayer(nn.Module):
    def __init__(self, remove_model_jit: bool = False, filt_size: int = 3, stride: int = 2,
                 channels: int = 0):
        super(AntiAliasDownsampleLayer, self).__init__()
        if not remove_model_jit:
            self.op = DownsampleJIT(filt_size, stride, channels)
        else:
            self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)


@torch.jit.script
class DownsampleJIT(object):
    def __init__(self, filt_size: int = 3, stride: int = 2, channels: int = 0):
        self.stride = stride
        self.filt_size = filt_size
        self.channels = channels

        assert self.filt_size == 3
        assert stride == 2
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).cuda().half()

    def __call__(self, input: torch.Tensor):
        if input.dtype != self.filt.dtype:
            self.filt = self.filt.float() 
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=2, padding=0, groups=input.shape[1])


class Downsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels


        assert self.filt_size == 3
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])


class bottleneck_head(nn.Module):
    def __init__(self, num_features, num_classes, bottleneck_features=200):
        super(bottleneck_head, self).__init__()
        self.embedding_generator = nn.ModuleList()
        self.embedding_generator.append(nn.Linear(num_features, bottleneck_features))
        self.embedding_generator = nn.Sequential(*self.embedding_generator)
        self.FC = nn.Linear(bottleneck_features, num_classes)

    def forward(self, x):
        self.embedding = self.embedding_generator(x)
        logits = self.FC(self.embedding)
        return logits


def conv2d(ni, nf, stride):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(nf),
        nn.ReLU(inplace=True)
    )


def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                  bias=False),
        InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param)
    )


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)

        out += residual

        out = self.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
                                        activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1,
                                                      activation="leaky_relu", activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
                                activation="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None: out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out


# class TResNet(Module):

#     def __init__(self, layers, in_chans=3, num_classes=1000, sep_features=0,width_factor=1.0,
#                  do_bottleneck_head=False,bottleneck_features=512):
#         super(TResNet, self).__init__()

#         # JIT layers
#         space_to_depth = SpaceToDepthModule()
#         anti_alias_layer = AntiAliasDownsampleLayer
#         global_pool_layer = FastAvgPool2d(flatten=True)

#         # TResnet stages
#         self.num_classes=num_classes
#         self.inplanes = int(64 * width_factor)
#         self.planes = int(64 * width_factor)
#         conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
#         layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 56x56
#         layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 28x28
#         layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 14x14
#         layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
#                                   anti_alias_layer=anti_alias_layer)  # 7x7

#         #self.globallayer=nn.Conv2d(2432,self.num_classes,8)
#         # body
#         self.base_model = nn.Sequential(OrderedDict([
#             ('SpaceToDepth', space_to_depth),
#             ('conv1', conv1),
#             ('layer1', layer1),
#             ('layer2', layer2),
#             ('layer3', layer3),
#             ('layer4', layer4)]))

#         # head
#         self.embeddings = []
#         self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
#         self.num_features = (self.planes * 8) * Bottleneck.expansion
#         if do_bottleneck_head:
#             fc = bottleneck_head(self.num_features, num_classes,
#                                  bottleneck_features=bottleneck_features)
#         else:
#             fc = nn.Linear(self.num_features , num_classes)

#         self.head = nn.Sequential(OrderedDict([('fc', fc)]))

#         # model initilization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         #comment following for common features
#         #self.class_feature_extractor=TRes_Feature_Extractor(vocab_size=num_classes,embedding_size=1024,global_feature_size=2432*8*8)#self.num_classes)#119168)

#         #uncomment following for common features
#         #print('Sep feautres:',sep_features)
#         print('Fatu extract:',sep_features)
#         if(sep_features==0):
#             self.finallayer=nn.utils.weight_norm(nn.Linear(2432*8*8,num_classes,bias=True))
#             self.forward_function=self.common_forward
#         else:
#             self.class_feature_extractor=TRes_Feature_Extractor(vocab_size=num_classes,embedding_size=1024,global_feature_size=2432*8*8)#self.num_classes)#119168)
#             #self.class_feature_extractor=TRes_Feature_Extractor(vocab_size=num_classes,embedding_size=1024,global_feature_size=2432*7*7)#self.num_classes)#119168)
#             self.forward_function=self.sep_forward



#         # residual connections special initialization
#         for m in self.modules():
#             if isinstance(m, BasicBlock):
#                 m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
#             if isinstance(m, Bottleneck):
#                 m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
#             if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

#     def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             layers = []
#             if stride == 2:
#                 # avg pooling before 1x1 conv
#                 layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
#             layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
#                                   activation="identity")]
#             downsample = nn.Sequential(*layers)

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
#                             anti_alias_layer=anti_alias_layer))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks): layers.append(
#             block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
#         return nn.Sequential(*layers)

#     def get_weights(self):
#         #return []
#         return self.class_feature_extractor.get_weights()

#     def sep_forward(self,x):
#         globalfeature = self.base_model(x)
#         globalfeature=torch.flatten(globalfeature,1)
#         allfeature=globalfeature.unsqueeze(dim=0).expand(self.num_classes,-1,-1)
#         allfeature=torch.flatten(allfeature,start_dim=0,end_dim=1)
#         indices=torch.arange(start=0,end=self.num_classes,step=1).view(-1,1)
#         #indices=torch.repeat_interleave(indices, repeats=10, dim=1)
#         #allindices=indices.repeat(x.shape[0],).long()
#         allindices=indices.expand(-1,x.shape[0]).flatten()
#         #print(allfeature.shape)

#         allout,out,features=self.class_feature_extractor(allfeature,allindices)

#         return out,features

#     def common_forward(self,x):
#         globalfeature = self.base_model(x)
#         globalfeature=torch.flatten(globalfeature,1)
#         #print(globalfeature.shape)
#         out=self.finallayer(globalfeature)
#         return out,globalfeature

#     def forward(self, x):

#         return self.forward_function(x)
#         # globalfeature = self.base_model(x)
#         # # print(globalfeature.shape)
#         # # 0/0
#         # #globalfeature=self.globallayer(globalfeature)
#         # globalfeature=torch.flatten(globalfeature,1)

#         # # allfeature=torch.repeat_interleave(globalfeature, repeats=self.num_classes, dim=0)
#         # allfeature=globalfeature.unsqueeze(dim=0).expand(self.num_classes,-1,-1)
        

#         # allfeature=torch.flatten(allfeature,start_dim=0,end_dim=1)
#         # indices=torch.arange(start=0,end=self.num_classes,step=1).view(-1,1)
#         # #indices=torch.repeat_interleave(indices, repeats=10, dim=1)
#         # #allindices=indices.repeat(x.shape[0],).long()
#         # allindices=indices.expand(-1,x.shape[0]).flatten()
#         # #print(allfeature.shape)

#         # allout,out,features=self.class_feature_extractor(allfeature,allindices)

#         # return out,features

#         # #######
        
#         # logits = self.head(self.embeddings)
#         # return logits,self.embeddings


# class TResNet(Module):

#     def __init__(self, layers, in_chans=3, num_classes=1000, sep_features=1,width_factor=1.0, first_two_layers=BasicBlock,
#                  do_bottleneck_head=False,bottleneck_features=512):
#         super(TResNet, self).__init__()

#         # JIT layers
#         space_to_depth = SpaceToDepthModule()
#         anti_alias_layer = AntiAliasDownsampleLayer
#         global_pool_layer = FastAvgPool2d(flatten=True)

#         # TResnet stages
#         self.num_classes=num_classes
#         self.inplanes = int(64 * width_factor)
#         self.planes = int(64 * width_factor)
#         conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)

#         layer1 = self._make_layer(first_two_layers, self.planes, layers[0], stride=1, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 56x56
#         layer2 = self._make_layer(first_two_layers, self.planes * 2, layers[1], stride=2, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 28x28

#         # layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
#         #                           anti_alias_layer=anti_alias_layer)  # 56x56
#         # layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
#         #                           anti_alias_layer=anti_alias_layer)  # 28x28

#         layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 14x14
#         layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
#                                   anti_alias_layer=anti_alias_layer)  # 7x7

#         #self.globallayer=nn.Conv2d(2432,self.num_classes,8)
#         # body

#         self.base_model = nn.Sequential(OrderedDict([
#             ('SpaceToDepth', space_to_depth),
#             ('conv1', conv1),
#             ('layer1', layer1),
#             ('layer2', layer2),
#             ('layer3', layer3),
#             ('layer4', layer4)]))

#         # head
#         self.embeddings = []
#         self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
#         self.num_features = (self.planes * 8) * Bottleneck.expansion
#         print('Says do bottle neck head:',do_bottleneck_head)
#         if do_bottleneck_head:
#             fc = bottleneck_head(self.num_features, num_classes,
#                                  bottleneck_features=bottleneck_features)
#         else:
#             fc = nn.Linear(self.num_features , num_classes)

#         self.head = nn.Sequential(OrderedDict([('fc', fc)]))

#         # model initilization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         #comment following for common features
#         #self.class_feature_extractor=TRes_Feature_Extractor(vocab_size=num_classes,embedding_size=1024,global_feature_size=2432*8*8)#self.num_classes)#119168)

#         #uncomment following for common features
#         #print('Sep feautres:',sep_features)

#         if(sep_features==0):
#             self.finallayer=nn.utils.weight_norm(nn.Linear(2432*8*8,num_classes,bias=True))
#             self.forward_function=self.common_forward
#         else:
#             self.class_feature_extractor=TRes_Feature_Extractor(vocab_size=num_classes,embedding_size=1024,global_feature_size=2432*7*7)#self.num_classes)#119168)
#             #self.class_feature_extractor=TRes_Feature_Extractor(vocab_size=num_classes,embedding_size=1024,global_feature_size=2432*8*8)#self.num_classes)#119168)
#             #self.class_feature_extractor=TRes_Feature_Extractor(vocab_size=num_classes,embedding_size=1024,global_feature_size=2432*14*14)#self.num_classes)#119168)
#             self.forward_function=self.sep_forward



#         # residual connections special initialization
#         for m in self.modules():
#             if isinstance(m, BasicBlock):
#                 m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
#             if isinstance(m, Bottleneck):
#                 m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
#             if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

#     def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             layers = []
#             if stride == 2:
#                 # avg pooling before 1x1 conv
#                 layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
#             layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
#                                   activation="identity")]
#             downsample = nn.Sequential(*layers)

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
#                             anti_alias_layer=anti_alias_layer))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks): layers.append(
#             block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
#         return nn.Sequential(*layers)

#     def set_inp(self,x):
#         pass 

#     def get_weights(self):
#         #return []
#         return self.class_feature_extractor.get_weights()

#     def sep_forward(self,x):
#         globalfeature = self.base_model(x)
#         #print('Global feature:',globalfeature.shape)
#         globalfeature=torch.flatten(globalfeature,1)

#         allfeature=globalfeature.unsqueeze(dim=0).expand(self.num_classes,-1,-1)
        
#         allfeature=torch.flatten(allfeature,start_dim=0,end_dim=1)
#         indices=torch.arange(start=0,end=self.num_classes,step=1).view(-1,1)
#         #indices=torch.repeat_interleave(indices, repeats=10, dim=1)
#         #allindices=indices.repeat(x.shape[0],).long()
#         allindices=indices.expand(-1,x.shape[0]).flatten()
#         #print(allfeature.shape)
#         allout,out,features=self.class_feature_extractor(allfeature,allindices)

#         return out,features

#     def common_forward(self,x):
#         globalfeature = self.base_model(x)

#         globalfeature=torch.flatten(globalfeature,1)
#         #print(globalfeature.shape)
#         out=self.finallayer(globalfeature)
#         return out,globalfeature

#     def forward(self, x,targetset=[]):

#         return self.forward_function(x)
#         # globalfeature = self.base_model(x)
#         # # print(globalfeature.shape)
#         # # 0/0
#         # #globalfeature=self.globallayer(globalfeature)
#         # globalfeature=torch.flatten(globalfeature,1)

#         # # allfeature=torch.repeat_interleave(globalfeature, repeats=self.num_classes, dim=0)
#         # allfeature=globalfeature.unsqueeze(dim=0).expand(self.num_classes,-1,-1)
        

#         # allfeature=torch.flatten(allfeature,start_dim=0,end_dim=1)
#         # indices=torch.arange(start=0,end=self.num_classes,step=1).view(-1,1)
#         # #indices=torch.repeat_interleave(indices, repeats=10, dim=1)
#         # #allindices=indices.repeat(x.shape[0],).long()
#         # allindices=indices.expand(-1,x.shape[0]).flatten()
#         # #print(allfeature.shape)

#         # allout,out,features=self.class_feature_extractor(allfeature,allindices)

#         # return out,features

#         # #######
        
#         # logits = self.head(self.embeddings)
#         # return logits,self.embeddings
        
        
        
# class TRes_Feature_Extractor(torch.nn.Module):
#     def __init__(self,vocab_size,embedding_size,global_feature_size):
#         super(TRes_Feature_Extractor,self).__init__()
#         self.embedding=nn.Embedding(vocab_size,embedding_size)
#         self.embedding_size=embedding_size
#         self.global_feature_size=global_feature_size
#         #self.linear1=nn.Linear(embedding_size+global_feature_size,2048)
#         self.linear1=nn.utils.weight_norm(nn.Linear(embedding_size+global_feature_size,2048),dim=0)
#         self.prelu1=nn.PReLU()
        
#         self.linear2=nn.utils.weight_norm(nn.Linear(2048,vocab_size),dim=0)
#         #self.linear2=nn.Linear(2048,vocab_size)
#         print(self.linear1,self.linear2)
#         self.vocab_size=vocab_size

#     def get_weights(self):
#         return [self.linear1,self.linear2]

#     def forward(self,features,embedding_idx):
#         emb=self.embedding(embedding_idx.cuda()).float()
#         #print(features.shape,emb.shape)
#         #print('Embedding:',emb.shape)
#         #print(features.shape,self.embedding_size,self.global_feature_size)
#         inputfeatures=torch.cat((features,emb),axis=-1)

#         #print('INpput features:',features.shape,emb.shape,inputfeatures.shape)
#         #print(self.linear1)
        
#         #print('Input features:',inputfeatures.shape)
#         #contrastivefeatures=self.normalize(self.linear1(inputfeatures))
#         features=self.prelu1(self.linear1(inputfeatures))

#         #print('Features:',features.shape)
#         alloutput=self.linear2(features)
#         #print(alloutput)
#         #print(embedding_idx)
#         output=alloutput[torch.arange(0,alloutput.shape[0]),embedding_idx]
#         #print(output)

#         output=output.reshape(self.vocab_size,-1).t()
        
#         #output=torch.reshape(output,(-1,self.vocab_size))
        
#         #features=torch.reshape(features,(batch_size,self.vocab_size,-1))
#         return alloutput,output,features


# class TResNet(Module):
#     def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0,
#                  do_bottleneck_head=False,bottleneck_features=512):
#         super(TResNet, self).__init__()

#         # JIT layers
#         space_to_depth = SpaceToDepthModule()
#         anti_alias_layer = AntiAliasDownsampleLayer
#         global_pool_layer = FastAvgPool2d(flatten=True)

#         # TResnet stages
#         self.inplanes = int(64 * width_factor)
#         self.planes = int(64 * width_factor)
#         conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
#         layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 56x56
#         layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 28x28
#         layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
#                                   anti_alias_layer=anti_alias_layer)  # 14x14
#         layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
#                                   anti_alias_layer=anti_alias_layer)  # 7x7

#         # body
#         self.body = nn.Sequential(OrderedDict([
#             ('SpaceToDepth', space_to_depth),
#             ('conv1', conv1),
#             ('layer1', layer1),
#             ('layer2', layer2),
#             ('layer3', layer3),
#             ('layer4', layer4)]))

#         # head
#         self.embeddings = []
#         self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
#         self.num_features = (self.planes * 8) * Bottleneck.expansion
#         if do_bottleneck_head:
#             fc = bottleneck_head(self.num_features, num_classes,
#                                  bottleneck_features=bottleneck_features)
#         else:
#             fc = nn.Linear(self.num_features , num_classes)

#         self.head = nn.Sequential(OrderedDict([('fc', fc)]))

#         # model initilization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # residual connections special initialization
#         for m in self.modules():
#             if isinstance(m, BasicBlock):
#                 m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
#             if isinstance(m, Bottleneck):
#                 m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
#             if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

#     def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             layers = []
#             if stride == 2:
#                 # avg pooling before 1x1 conv
#                 layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
#             layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
#                                   activation="identity")]
#             downsample = nn.Sequential(*layers)

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
#                             anti_alias_layer=anti_alias_layer))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks): layers.append(
#             block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
#         return nn.Sequential(*layers)

#     def get_weights(self):
#         return []

#     def forward(self, x):
#         x = self.body(x)
#         self.embeddings = self.global_pool(x)
#         logits = self.head(self.embeddings)
#         return logits,None

        
        
        
# class TRes_Feature_Extractor(torch.nn.Module):
#     def __init__(self,vocab_size,embedding_size,global_feature_size):
#         super(TRes_Feature_Extractor,self).__init__()
#         self.embedding=nn.Embedding(vocab_size,embedding_size)
#         #self.linear1=nn.Linear(embedding_size+global_feature_size,2048)
#         self.linear1=nn.utils.weight_norm(nn.Linear(embedding_size+global_feature_size,2048),dim=0)
#         self.prelu1=nn.PReLU()
        
#         self.linear2=nn.utils.weight_norm(nn.Linear(2048,vocab_size),dim=0)
#         #self.linear2=nn.Linear(2048,vocab_size)
#         print(self.linear1,self.linear2)
#         self.vocab_size=vocab_size

#     def get_weights(self):
#         return [self.linear1,self.linear2]

#     def forward(self,features,embedding_idx):
        

#         emb=self.embedding(embedding_idx.cuda()).float()
#         #print(features.shape,emb.shape)
#         #print('Embedding:',emb.shape)
#         inputfeatures=torch.cat((features,emb),axis=-1)
        
        
#         #print('Input features:',inputfeatures.shape)
#         #contrastivefeatures=self.normalize(self.linear1(inputfeatures))
#         features=self.prelu1(self.linear1(inputfeatures))

#         #print('Features:',features.shape)
#         alloutput=self.linear2(features)
#         #print(alloutput)
#         #print(embedding_idx)
#         output=alloutput[torch.arange(0,alloutput.shape[0]),embedding_idx]
#         #print(output)

#         output=output.reshape(self.vocab_size,-1).t()
        
#         #output=torch.reshape(output,(-1,self.vocab_size))
        
#         #features=torch.reshape(features,(batch_size,self.vocab_size,-1))
#         return alloutput,output,features
#         #return output,features


# def TResnetM(model_params):
#     """Constructs a medium TResnet model.
#     """
#     in_chans = 3
#     num_classes = model_params['num_classes']
#     model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans)
#     return model


# # def TResnetL(model_params):
# #     """Constructs a large TResnet model.
# #     """
# #     in_chans = 3
# #     num_classes = model_params['num_classes']
# #     #do_bottleneck_head = model_params['args'].do_bottleneck_head
# #     do_bottleneck_head = False
# #     model = TResNet(layers=[4, 5, 18, 3], num_classes=num_classes,sep_features=model_params['sep_features'], in_chans=in_chans, width_factor=1.2,
# #                     do_bottleneck_head=do_bottleneck_head)
# #     return model

# def TResnetMLDecoder(model_params):
#     """Constructs a large TResnet model.
#     """
#     in_chans = 3
#     num_classes = model_params['num_classes']
#     layers_list = [3, 4, 23, 3]
#     model = TResNet(layers=layers_list, num_classes=num_classes, in_chans=in_chans, first_two_layers=Bottleneck)
#     return model

# def TResnetL(model_params):
#     """Constructs a large TResnet model.
#     """
#     in_chans = 3
#     num_classes = model_params['num_classes']
#     do_bottleneck_head = model_params['args'].do_bottleneck_head
#     # do_bottleneck_head = True
#     model = TResNet(layers=[4, 5, 18, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.2,
#                     do_bottleneck_head=do_bottleneck_head)
#     return model


# def TResnetXL(model_params):
#     """Constructs a xlarge TResnet model.
#     """
#     in_chans = 3
#     num_classes = model_params['num_classes']
#     model = TResNet(layers=[4, 5, 24, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.3)

#     return model


# ######################################

# FOR non open images, comment below and uncomment above


import torch
import torch.nn as nn
from torch.nn import Module as Module
from collections import OrderedDict
from src.models.tresnet.layers.anti_aliasing import AntiAliasDownsampleLayer
from .layers.avg_pool import FastAvgPool2d
from .layers.general_layers import SEModule, SpaceToDepthModule
from inplace_abn import InPlaceABN, ABN
import numpy as np 
import random 
import os 

# seed=999
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def InplacABN_to_ABN(module: nn.Module) -> nn.Module:
    # convert all InplaceABN layer to bit-accurate ABN layers.
    if isinstance(module, InPlaceABN):
        module_new = ABN(module.num_features, activation=module.activation,
                         activation_param=module.activation_param)
        for key in module.state_dict():
            module_new.state_dict()[key].copy_(module.state_dict()[key])
        module_new.training = module.training
        module_new.weight.data = module_new.weight.abs() + module_new.eps
        return module_new
    for name, child in reversed(module._modules.items()):
        new_child = InplacABN_to_ABN(child)
        if new_child != child:
            module._modules[name] = new_child
    return module

class bottleneck_head(nn.Module):
    def __init__(self, num_features, num_classes, bottleneck_features=200):
        super(bottleneck_head, self).__init__()
        self.embedding_generator = nn.ModuleList()
        self.embedding_generator.append(nn.Linear(num_features, bottleneck_features))
        self.embedding_generator = nn.Sequential(*self.embedding_generator)
        self.FC = nn.Linear(bottleneck_features, num_classes)

    def forward(self, x):
        self.embedding = self.embedding_generator(x)
        logits = self.FC(self.embedding)
        return logits


def conv2d(ni, nf, stride):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(nf),
        nn.ReLU(inplace=True)
    )


def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                  bias=False),
        InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param)
    )


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)

        out += residual

        out = self.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
                                        activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1,
                                                      activation="leaky_relu", activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
                                activation="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None: out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out

import pickle 
class TResNet(Module):

    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0,
                 do_bottleneck_head=False,bottleneck_features=512):
        super(TResNet, self).__init__()

        # JIT layers
        self.out_to_hierarchy=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/out_to_hierarchy','rb'))
        self.output_indices=torch.tensor(list(self.out_to_hierarchy.keys()))
        self.hier_out_keys=torch.tensor(list(self.out_to_hierarchy.values()))

        space_to_depth = SpaceToDepthModule()
        anti_alias_layer = AntiAliasDownsampleLayer
        global_pool_layer = FastAvgPool2d(flatten=True)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 56x56
        layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 28x28
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 14x14
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
                                  anti_alias_layer=anti_alias_layer)  # 7x7

        # body
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))

        # head
        self.embeddings = []
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        if do_bottleneck_head:
            fc = bottleneck_head(self.num_features, num_classes,
                                 bottleneck_features=bottleneck_features)
        else:
            fc = nn.Linear(self.num_features , num_classes)

        self.head = nn.Sequential(OrderedDict([('fc', fc)]))

        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                                  activation="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(
            block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x,par,targetset=[]):
        #x=torch.clamp(x,0.0,1.0)
        #out = self.body(x)

        out0=self.body[0:1](x)
        out1=self.body[1:2](out0)
        out2=self.body[2:3](out1)
        out3=self.body[3:4](out2) #has issue
        out4=self.body[4:5](out3) #has issue
        out=self.body[5:6](out4)

        # print('First:\n\n',self.body[2:3])
        # print('Second:\n\n',self.body[3:4])
        # print('Third:\n\n',self.body[4:5])
        # print('Fourth:\n\n',self.body[5:6])

        # r=torch.clone(out2).detach()
        # r.requires_grad=True 
        # for _ in range(20):
        #     y=self.body[3:4](r)
        #     print(torch.sum(r),torch.sum(y),torch.autograd.grad(y.sum(), [r])[0].detach().sum())

        
        # 0/0


        
        self.embeddings = self.global_pool(out)
        logits = self.head(self.embeddings)
        # print(torch.autograd.grad(torch.sum(logits),x))
        # 0/0
        
        #hier_logits=torch.zeros(size=(logits.shape[0],601),dtype=torch.float32).cuda()
        #hier_logits[:,self.hier_out_keys]=logits[:,self.output_indices]
        hier_logits=logits 

        #return logits, out 
        return hier_logits, out 


# def TResnetM(model_params):
#     """Constructs a medium TResnet model.
#     """
#     in_chans = 3
#     num_classes = model_params['num_classes']
#     model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans)
#     return model

def TResnetM(model_params):
    """Constructs a medium TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    #print('Do bottle:',model_params['args'].do_bottleneck_head)
    do_bottleneck_head = model_params['args'].do_bottleneck_head
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, do_bottleneck_head=do_bottleneck_head, in_chans=in_chans)
    return model


def TResnetL(model_params):
    """Constructs a large TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    do_bottleneck_head = model_params['args'].do_bottleneck_head

    model = TResNet(layers=[4, 5, 18, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.2,
                    do_bottleneck_head=do_bottleneck_head)
    return model


def TResnetXL(model_params):
    """Constructs a xlarge TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    model = TResNet(layers=[4, 5, 24, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.3)

    return model

# ########################################
# #for lvis


# # import sys
# # import os
# # sys.path.append('./')
# # import torch
# # import torch.nn as nn
# # from torch.nn import Module as Module
# # from collections import OrderedDict
# # #from .layers.anti_aliasing import AntiAliasDownsampleLayer
# # #from .layers.avg_pool import FastAvgPool2d
# # #from .layers.general_layers import SEModule, SpaceToDepthModule
# # from inplace_abn import InPlaceABN
# # import torch.nn.parallel
# # import numpy as np
# # import torch.nn.functional as F


# # class Flatten(nn.Module):
# #     def forward(self, x):
# #         return x.view(x.size(0), -1)


# # class DepthToSpace(nn.Module):

# #     def __init__(self, block_size):
# #         super().__init__()
# #         self.bs = block_size

# #     def forward(self, x):
# #         N, C, H, W = x.size()
# #         x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
# #         x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
# #         x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
# #         return x


# # class SpaceToDepthModule(nn.Module):
# #     def __init__(self, remove_model_jit=False):
# #         super().__init__()
# #         if not remove_model_jit:
# #             self.op = SpaceToDepthJit()
# #         else:
# #             self.op = SpaceToDepth()

# #     def forward(self, x):
# #         return self.op(x)


# # class SpaceToDepth(nn.Module):
# #     def __init__(self, block_size=4):
# #         super().__init__()
# #         assert block_size == 4
# #         self.bs = block_size

# #     def forward(self, x):
# #         N, C, H, W = x.size()
# #         x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
# #         x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
# #         x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
# #         return x


# # @torch.jit.script
# # class SpaceToDepthJit(object):
# #     def __call__(self, x: torch.Tensor):
# #         # assuming hard-coded that block_size==4 for acceleration
# #         N, C, H, W = x.size()
# #         x = x.view(N, C, H // 4, 4, W // 4, 4)  # (N, C, H//bs, bs, W//bs, bs)
# #         x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
# #         x = x.view(N, C * 16, H // 4, W // 4)  # (N, C*bs^2, H//bs, W//bs)
# #         return x


# # class hard_sigmoid(nn.Module):
# #     def __init__(self, inplace=True):
# #         super(hard_sigmoid, self).__init__()
# #         self.inplace = inplace

# #     def forward(self, x):
# #         if self.inplace:
# #             return x.add_(3.).clamp_(0., 6.).div_(6.)
# #         else:
# #             return F.relu6(x + 3.) / 6.


# # class SEModule(nn.Module):

# #     def __init__(self, channels, reduction_channels, inplace=True):
# #         super(SEModule, self).__init__()
# #         self.avg_pool = FastAvgPool2d()
# #         self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, padding=0, bias=True)
# #         self.relu = nn.ReLU(inplace=inplace)
# #         self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
# #         # self.activation = hard_sigmoid(inplace=inplace)
# #         self.activation = nn.Sigmoid()

# #     def forward(self, x):
# #         x_se = self.avg_pool(x)
# #         x_se2 = self.fc1(x_se)
# #         x_se2 = self.relu(x_se2)
# #         x_se = self.fc2(x_se2)
# #         x_se = self.activation(x_se)
# #         return x * x_se


# # class FastAvgPool2d(nn.Module):
# #     def __init__(self, flatten=False):
# #         super(FastAvgPool2d, self).__init__()
# #         self.flatten = flatten

# #     def forward(self, x):
# #         if self.flatten:
# #             in_size = x.size()
# #             return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
# #         else:
# #             return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)




# # class AntiAliasDownsampleLayer(nn.Module):
# #     def __init__(self, remove_model_jit: bool = False, filt_size: int = 3, stride: int = 2,
# #                  channels: int = 0):
# #         super(AntiAliasDownsampleLayer, self).__init__()
# #         if not remove_model_jit:
# #             self.op = DownsampleJIT(filt_size, stride, channels)
# #         else:
# #             self.op = Downsample(filt_size, stride, channels)

# #     def forward(self, x):
# #         return self.op(x)


# # @torch.jit.script
# # class DownsampleJIT(object):
# #     def __init__(self, filt_size: int = 3, stride: int = 2, channels: int = 0):
# #         self.stride = stride
# #         self.filt_size = filt_size
# #         self.channels = channels

# #         assert self.filt_size == 3
# #         assert stride == 2
# #         a = torch.tensor([1., 2., 1.])

# #         filt = (a[:, None] * a[None, :]).clone().detach()
# #         filt = filt / torch.sum(filt)
# #         self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).cuda().half()

# #     def __call__(self, input: torch.Tensor):
# #         if input.dtype != self.filt.dtype:
# #             self.filt = self.filt.float() 
# #         input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
# #         return F.conv2d(input_pad, self.filt, stride=2, padding=0, groups=input.shape[1])


# # class Downsample(nn.Module):
# #     def __init__(self, filt_size=3, stride=2, channels=None):
# #         super(Downsample, self).__init__()
# #         self.filt_size = filt_size
# #         self.stride = stride
# #         self.channels = channels


# #         assert self.filt_size == 3
# #         a = torch.tensor([1., 2., 1.])

# #         filt = (a[:, None] * a[None, :]).clone().detach()
# #         filt = filt / torch.sum(filt)
# #         self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))

# #     def forward(self, input):
# #         input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
# #         return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])




# # class bottleneck_head(nn.Module):
# #     def __init__(self, num_features, num_classes, bottleneck_features=200):
# #         super(bottleneck_head, self).__init__()
# #         self.embedding_generator = nn.ModuleList()
# #         self.embedding_generator.append(nn.Linear(num_features, bottleneck_features))
# #         self.embedding_generator = nn.Sequential(*self.embedding_generator)
# #         self.FC = nn.Linear(bottleneck_features, num_classes)

# #     def forward(self, x):
# #         self.embedding = self.embedding_generator(x)
# #         logits = self.FC(self.embedding)
# #         return logits


# # def conv2d(ni, nf, stride):
# #     return nn.Sequential(
# #         nn.Conv2d(ni, nf, kernel_size=3, stride=stride, padding=1, bias=False),
# #         nn.BatchNorm2d(nf),
# #         nn.ReLU(inplace=True)
# #     )


# # def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
# #     return nn.Sequential(
# #         nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
# #                   bias=False),
# #         InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param)
# #     )


# # class BasicBlock(Module):
# #     expansion = 1

# #     def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
# #         super(BasicBlock, self).__init__()
# #         if stride == 1:
# #             self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
# #         else:
# #             if anti_alias_layer is None:
# #                 self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
# #             else:
# #                 self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
# #                                            anti_alias_layer(channels=planes, filt_size=3, stride=2))

# #         self.conv2 = conv2d_ABN(planes, planes, stride=1, activation="identity")
# #         self.relu = nn.ReLU(inplace=True)
# #         self.downsample = downsample
# #         self.stride = stride
# #         reduce_layer_planes = max(planes * self.expansion // 4, 64)
# #         self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

# #     def forward(self, x):
# #         if self.downsample is not None:
# #             residual = self.downsample(x)
# #         else:
# #             residual = x

# #         out = self.conv1(x)
# #         out = self.conv2(out)

# #         if self.se is not None: out = self.se(out)

# #         out += residual

# #         out = self.relu(out)

# #         return out


# # class Bottleneck(Module):
# #     expansion = 4

# #     def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
# #         super(Bottleneck, self).__init__()
# #         self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
# #                                 activation_param=1e-3)
# #         if stride == 1:
# #             self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
# #                                     activation_param=1e-3)
# #         else:
# #             if anti_alias_layer is None:
# #                 self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
# #                                         activation_param=1e-3)
# #             else:
# #                 self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1,
# #                                                       activation="leaky_relu", activation_param=1e-3),
# #                                            anti_alias_layer(channels=planes, filt_size=3, stride=2))

# #         self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
# #                                 activation="identity")

# #         self.relu = nn.ReLU(inplace=True)
# #         self.downsample = downsample
# #         self.stride = stride

# #         reduce_layer_planes = max(planes * self.expansion // 8, 64)
# #         self.se = SEModule(planes, reduce_layer_planes) if use_se else None

# #     def forward(self, x):
# #         if self.downsample is not None:
# #             residual = self.downsample(x)
# #         else:
# #             residual = x

# #         out = self.conv1(x)
# #         out = self.conv2(out)
# #         if self.se is not None: out = self.se(out)

# #         out = self.conv3(out)
# #         out = out + residual  # no inplace
# #         out = self.relu(out)

# #         return out


# # class TResNet(Module):

# #     def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0,
# #                  do_bottleneck_head=False, bottleneck_features=512, first_two_layers=BasicBlock):
# #         super(TResNet, self).__init__()

# #         # JIT layers
# #         space_to_depth = SpaceToDepthModule()
# #         anti_alias_layer = AntiAliasDownsampleLayer
# #         global_pool_layer = FastAvgPool2d(flatten=True)

# #         # TResnet stages
# #         self.inplanes = int(64 * width_factor)
# #         self.planes = int(64 * width_factor)
# #         conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
# #         layer1 = self._make_layer(first_two_layers, self.planes, layers[0], stride=1, use_se=True,
# #                                   anti_alias_layer=anti_alias_layer)  # 56x56
# #         layer2 = self._make_layer(first_two_layers, self.planes * 2, layers[1], stride=2, use_se=True,
# #                                   anti_alias_layer=anti_alias_layer)  # 28x28
# #         layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
# #                                   anti_alias_layer=anti_alias_layer)  # 14x14
# #         layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
# #                                   anti_alias_layer=anti_alias_layer)  # 7x7

# #         # body
# #         self.body = nn.Sequential(OrderedDict([
# #             ('SpaceToDepth', space_to_depth),
# #             ('conv1', conv1),
# #             ('layer1', layer1),
# #             ('layer2', layer2),
# #             ('layer3', layer3),
# #             ('layer4', layer4)]))

# #         # head
# #         self.embeddings = []
# #         self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
# #         self.num_features = (self.planes * 8) * Bottleneck.expansion
# #         if do_bottleneck_head:
# #             fc = bottleneck_head(self.num_features, num_classes,
# #                                  bottleneck_features=bottleneck_features)
# #         else:
# #             fc = nn.Linear(self.num_features, num_classes)

# #         self.head = nn.Sequential(OrderedDict([('fc', fc)]))

# #         # model initilization
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
# #             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
# #                 nn.init.constant_(m.weight, 1)
# #                 nn.init.constant_(m.bias, 0)

# #         # residual connections special initialization
# #         for m in self.modules():
# #             if isinstance(m, BasicBlock):
# #                 m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
# #             if isinstance(m, Bottleneck):
# #                 m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
# #             if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

# #     def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
# #         downsample = None
# #         if stride != 1 or self.inplanes != planes * block.expansion:
# #             layers = []
# #             if stride == 2:
# #                 # avg pooling before 1x1 conv
# #                 layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
# #             layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
# #                                   activation="identity")]
# #             downsample = nn.Sequential(*layers)

# #         layers = []
# #         layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
# #                             anti_alias_layer=anti_alias_layer))
# #         self.inplanes = planes * block.expansion
# #         for i in range(1, blocks): layers.append(
# #             block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
# #         return nn.Sequential(*layers)

# #     def forward(self, x,par,targetset=[]):
# #         x = self.body(x)
# #         self.embeddings = self.global_pool(x)
# #         logits = self.head(self.embeddings)
# #         return logits,self.embeddings


# # def TResnetM(model_params):
# #     """Constructs a medium TResnet model.
# #     """
# #     in_chans = 3
# #     num_classes = model_params['num_classes']
# #     model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans)
# #     return model


# # def TResnetL(model_params):
# #     """Constructs a large TResnet model.
# #     """
# #     in_chans = 3
# #     num_classes = model_params['num_classes']
# #     layers_list = [3, 4, 23, 3]
# #     model = TResNet(layers=layers_list, num_classes=num_classes, in_chans=in_chans, first_two_layers=Bottleneck)
# #     return model
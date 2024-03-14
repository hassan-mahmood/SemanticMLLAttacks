import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic=True 
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
        filt = filt / (torch.sum(filt)+1e-9)    #hassan change
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).cuda().half()

    def __call__(self, input: torch.Tensor):
        if input.dtype != self.filt.dtype:
            self.filt = self.filt.float() 
        #input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        with torch.no_grad():
            input_pad = torch.clone(F.pad(input, (1, 1, 1, 1), 'reflect')).detach()
            
        input_pad[:,:,1:-1,1:-1]=input

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
        filt = filt / (torch.sum(filt)+1e-9)    #hassan change
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))

    def forward(self, input):
        #input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        
        with torch.no_grad():
            input_pad = torch.clone(F.pad(input, (1, 1, 1, 1), 'reflect')).detach()
            
        input_pad[:,:,1:-1,1:-1]=input

        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])

    # def forward(self, input):
    #     input=input.cuda()
    #     # input_pad = F.pad(input, (1, 1, 1, 1), 'reflect').cuda()
    #     r=torch.clone(input).detach().cuda()
    #     r.requires_grad=True 
        
        
    #     for _ in range(20):
    #         input_pad = F.pad(r, (1, 1, 1, 1), 'reflect').cuda()
    #         y=F.conv2d(input_pad.cuda(), self.filt.cuda(), stride=self.stride, padding=0, groups=input.shape[1])
    #         print(torch.autograd.grad(y.sum(), [r])[0].detach().sum())

        
        
    #     return F.conv2d(input_pad.cuda(), self.filt.cuda(), stride=self.stride, padding=0, groups=input.shape[1])



import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from utils.options import args
import time

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        w = self.weight
        self.alpha = nn.Parameter(w.abs().mean([1,2,3], keepdim=True).view(w.size(0), 1 ,1), requires_grad=True)
        B = torch.sign(w).view(w.shape[0], 1, -1)
        B.add_(1).float().div_(2)
        self.B = nn.Parameter(B, requires_grad=False)
        percent = torch.tensor(0.)
        self.register_buffer('percent', percent) 

    def forward(self, input):
        a = input
        w = self.weight
        device = w.device
        percent = self.percent.to(device)
        w1 = (w - w.mean([1,2,3], keepdim=True)).detach()
        clip_L = np.percentile(w1.abs().flatten().cpu(), percent.item())
        clip_H = np.percentile(w1.abs().flatten().cpu(), 100-percent.item())
        self.weight.data = w1.abs().clamp(min=clip_L,max=clip_H).mul(torch.sign(w1))
        del w, w1, clip_L, clip_H
        w = self.weight.to(device)
        w1 = w - w.mean([1,2,3], keepdim=True)
        w2 = w1 / torch.sqrt(w1.var([1,2,3], keepdim=True) + 1e-5)
        
        a0 = a - a.mean([1,2,3], keepdim=True)
        a1 = a0 / torch.sqrt(a0.var([1,2,3], keepdim=True) + 1e-5)
        X = w2.abs().view(w.shape[0], -1)
        if self.training:
            #* update B
            if args.sort == 'quick':
                B_tilde = []
                for i in range(self.B.shape[0]): 
                    b = torch.zeros_like(self.B[0].squeeze()).to(device)
                    Xi = (X[i].detach())
                    max_item, max_arg = torch.sort(Xi, descending=True) 
                    b[max_arg[0:len(max_item)//2]] = 1 
                    B_tilde.append(b)
                    del max_item, max_arg, b, Xi
                self.B.data = torch.stack(B_tilde).unsqueeze(1)
                del B_tilde
            else:  # median
                B = torch.zeros_like(X).cuda()
                B[X > X.median(dim=1)[0].unsqueeze(1)] = 1 
                self.B.data = B
                del B

        X2 = X.view_as(w)
        bw = X2 - X2.detach() + self.B.mul(2).sub(1).view_as(w).to(device)
        ba = BinaryQuantize().apply(a1)
        #* 1bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha

        return output


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input
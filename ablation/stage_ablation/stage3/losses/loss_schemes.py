#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange as o_rearrange
from einops.layers.torch import Rearrange

def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()


class MultiTaskLoss(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.l1 = nn.L1Loss(size_average=True, reduce=True)

    
    def forward(self, pred, gt, tasks):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}

        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in tasks]))

        if self.p.intermediate_supervision:
            inter_preds = pred['inter_preds']
            losses_inter = {t: self.loss_ft[t](inter_preds[t], gt[t]) for t in self.tasks}
            for k, v in losses_inter.items():
                out['inter_%s' %(k)] = v
                out['total'] += self.loss_weights[k] * v #* 0.5
        # 计算一下深度注意力机制的损失
        # dav = pred['dav']
        # gt_dav = self.compute_DAV(gt['depth'])
        # l1 = self.l1(dav,gt_dav)
        # out['total']+=l1
        return out
    # 计算真实值的dav
    def compute_DAV(self,x):
        # 为了防止出现超出gpu的首先对真实值进行下采样
        x = F.interpolate(x, size=(14,18), mode="bilinear", align_corners=False)
        green2 = rearrange(x, 'b c h w -> b (h w) c')
        blue2 = rearrange(x, 'b c h w -> b c (h w)')
        dav_1 = torch.einsum('bic,bcj->bij', (green2, blue2))  # will need to be reshaped for this operation
        dav_2 = torch.sigmoid(dav_1)
        return dav_2
        pass


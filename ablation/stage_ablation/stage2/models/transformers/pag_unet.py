# By Changhao Li
import torch
import torch.nn as nn
from einops import rearrange as o_rearrange
from models.transformers.fin_decoder import FinalDecoder
import torch.nn.functional as F
INTERPOLATE_MODE = 'bilinear'
BATCHNORM = nn.BatchNorm2d

def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()

def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()


class PagUnet(nn.Module):
    def __init__(self, p):
        super(PagUnet, self).__init__()
        self.embed_dim = p.embed_dim
        p.mtt_resolution = [_ for _ in p.spatial_dim[-1]]
        self.p = p
        self.all_tasks = p.TASKS.NAMES
        task_no = len(self.all_tasks)
        self.task_no = task_no
        embed_dim_with_pred = self.embed_dim + p.PRED_OUT_NUM_CONSTANT
        self.scale_embed = nn.ModuleList()
        #  2-stage
        self.scale_embed.append(nn.Conv2d(p.backbone_channels[2], embed_dim_with_pred, 3, padding=1))
        self.scale_embed.append(None)
        # 初步解码器
        self.pre_decoder = Per_decoder(p)
        # 最终的解码器
        self.fin_decoer = FinalDecoder(p, in_chans=embed_dim_with_pred)

        pass

    def forward(self, x_list):
        ori_feature = x_list[-1]
        # 0. 对每阶段的信息进行处理
        back_fea = []
        for sca in range(len(x_list)):
            oh, ow = self.p.spatial_dim[-1]
            _fea = x_list[sca]
            _fea = rearrange(_fea, 'b (h w) c -> b c h w', h=oh, w=ow)
            if sca == 1:
                x = _fea
            if self.scale_embed[sca] != None:
                _fea = self.scale_embed[sca](_fea)
            back_fea.append(_fea)
            pass
        # 1. 获得每个任务的特征
        h, w = self.p.mtt_resolution
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        ms_feat_dict, inter_pred = self.pre_decoder(x)
        x_dict = self.fin_decoer(ms_feat_dict, inter_pred, back_fea, ori_feature)
        return x_dict, inter_pred
        pass


class Per_decoder(nn.Module):
    def __init__(self, p):
        super(Per_decoder, self).__init__()
        self.p = p
        self.embed_dim = p.embed_dim
        input_channels = p.backbone_channels[-1]
        task_channels = self.embed_dim

        self.intermediate_head = nn.ModuleDict()
        self.preliminary_decoder = nn.ModuleDict()

        for t in p.TASKS.NAMES:
            self.intermediate_head[t] = nn.Conv2d(task_channels, p.TASKS.NUM_OUTPUT[t], 1)
            self.preliminary_decoder[t] = nn.Sequential(
                    ConvBlock(input_channels, input_channels),
                    ConvBlock(input_channels, task_channels),
                )
        pass

    def forward(self, x):
        ms_feat_dict = {}
        inter_pred = {}
        for task in self.p.TASKS.NAMES:
            _x = self.preliminary_decoder[task](x)
            ms_feat_dict[task] = _x
            _inter_p = self.intermediate_head[task](_x)
            inter_pred[task] = _inter_p
            pass
        return ms_feat_dict, inter_pred
        pass


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BATCHNORM
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out
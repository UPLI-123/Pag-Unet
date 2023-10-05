# by LCH
import torch
import torch.nn as nn
from einops import rearrange as o_rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_
from collections import OrderedDict
import torch.nn.functional as F
BATCHNORM = nn.BatchNorm2d
bn_mom = 0.1
algc = False
INTERPOLATE_MODE = 'bilinear'


def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()


class FinalDecoder(nn.Module):
    def __init__(self, p, in_chans):
        super(FinalDecoder, self).__init__()
        self.p = p
        self.all_tasks = p.TASKS.NAMES
        task_no = len(self.all_tasks)
        self.task_no = task_no
        embed_dim = in_chans
        self.embed_dim = embed_dim
        # 融合信息
        self.mix_proj = nn.ModuleDict()
        ori_embed_dim = 512
        for t in self.all_tasks:
            _mix_channel = ori_embed_dim + p.TASKS.NUM_OUTPUT[t]
            # print(embed_dim)
            self.mix_proj[t] = nn.Sequential(nn.Conv2d(_mix_channel, embed_dim, 1))
            pass

        # 自注意力机制所要使用的参数信息
        spec = {
            'DIM_EMBED': [embed_dim // 2, embed_dim // 4, embed_dim // 8],
            'NUM_HEADS': [2, 2, 2],
            'MLP_RATIO': [4., 4., 4.],
            'DROP_PATH_RATE': [0.15, 0.15, 0.15],
            'QKV_BIAS': [True, True, True],
            'KV_PROJ_METHOD': ['avg', 'avg', 'avg'],
            'KERNEL_KV': [4, 8, 16],
            'PADDING_KV': [0, 0, 0],
            'STRIDE_KV': [4, 8, 16],
            'Q_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
            'KERNEL_Q': [3, 3, 3],
            'PADDING_Q': [1, 1, 1],
            'STRIDE_Q': [2, 2, 2],
        }


        self.update1 = nn.ModuleList([])
        self.update2 = nn.ModuleList([])
        self.mix_res = nn.ModuleDict()
        self.fin_cov = nn.ModuleList([])
        for i, task in enumerate(self.all_tasks):
            update1 = ConvBlock(embed_dim, embed_dim // 4)
            update2 = ConvBlock(embed_dim // 2, embed_dim // 4)
            self.mix_res[task] = nn.Sequential(
                ConvBlock(embed_dim // 4, embed_dim // 8),
                ConvBlock(embed_dim // 8, embed_dim // 4)
            )
            self.update1.append(update1)
            self.update2.append(update2)
            cov1 = ConvBlock(1024, 512)
            self.fin_cov.append(cov1)
            pass

        pass
    
    def forward(self, x_dict, inter_pred, back_fea, ori_feature):
        # 输入信息的产生
        x_list = []
        for task in self.all_tasks:
            _x = x_dict[task]
            _x = torch.cat([_x, inter_pred[task]], dim=1)
            _x = self.mix_proj[task](_x)
            x_list.append(_x)
            pass
        # print(x_list)
        ecb40 = x_list
        oh, ow = self.p.spatial_dim[-1]
        ori_feature = rearrange(ori_feature, 'b (h w) c -> b c h w', h=oh, w=ow)
        for i, task in enumerate(self.all_tasks):
            h, w = ecb40[0].shape[2:]
            if i == 0:
                ori_feature = F.interpolate(ori_feature, size=(h, w), mode='bilinear', align_corners=False)
                pass
            task_feature = self.fin_cov[i](ori_feature)
            x_dict[task] = x_dict[task] + task_feature
        return x_dict

        pass



# 多层感知机
class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    pass
# 卷积操作
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
# 局部特征提取模块,实际上就是两个卷积用来提取局部信息
class Part_feature(nn.Module):
    def __init__(self, in_dim):
        super(Part_feature, self).__init__()
        self.cov1 = ConvBlock(in_dim, in_dim // 2)
        self.cov2 = ConvBlock(in_dim//2, in_dim)
        pass

    def forward(self, x):
        out = self.cov1(x)
        out = self.cov2(out)
        return out
        pass
#  两特征融合函数
class Mix_feature(nn.Module):
    def __init__(self):
        super(Mix_feature, self).__init__()
        pass

    def forward(self, x, y):
        '''
        :param x:  自注意获取的特征
        :param y:  任务的补充的信息
        :return:
        '''
        total = x*y
        sim_map = torch.sigmoid(total)
        x = sim_map*x + (1-sim_map)*y
        return x
        pass
class Pag(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(Pag, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)
        x = (1 - sim_map) * x + sim_map * y
        return x

        pass



# by Li Chang Hao
#  主要实现Pag_Unet的初步编码器代码
import torch
import torch.nn as nn
from models.PagUnet.fin_decoder import Fin_Decoder
INTERPOLATE_MODE = 'bilinear'
BATCHNORM = nn.BatchNorm2d

class PagUnet_PreDecoder(nn.Module):
    def __init__(self, p):
        super(PagUnet_PreDecoder, self).__init__()
        self.p = p
        # 1. 初步解码器
        self.pre_decoder = Per_decoder(p)
        embed_dim_with_pred = 2048
        # 2. 最终的解码器Pag_Unet
        self.fin_decoder = Fin_Decoder(p, in_chans=embed_dim_with_pred)
        pass

    def forward(self, x_list):
        '''
        :param x_list:
        :return:
        '''
        ori_feature = x_list[-1]
        ms_feat_dict, inter_pred = self.pre_decoder(ori_feature)
        x_dict = self.fin_decoder(ms_feat_dict, inter_pred, x_list, ori_feature)
        return x_dict, inter_pred
        pass
    pass


class Per_decoder(nn.Module):
    def __init__(self, p):
        super(Per_decoder, self).__init__()
        self.p = p
        self.embed_dim = p.embed_dim
        input_channels = 2048
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
#  最终的编码器
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
            'DIM_EMBED': [embed_dim // 2, embed_dim//4, embed_dim//8],
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

        # 获取dcb31
        self.Pself1 = PselfAttention(p, spec, 0, embed_dim, task_no)
        self.Pself2 = PselfAttention(p, spec, 1, embed_dim // 2, task_no)
        self.Pself3 = PselfAttention(p, spec, 2, embed_dim // 4, task_no)

        self.pag1 = Pag(embed_dim // 2, embed_dim // 4)
        self.pag2 = Pag(embed_dim // 4, embed_dim // 8)
        self.pag3 = Pag(embed_dim // 4, embed_dim // 8)

        self.compress1 = ConvBlock(embed_dim, embed_dim // 2)
        self.compress2 = ConvBlock(embed_dim // 2, embed_dim // 4)
        self.compress3 = ConvBlock(embed_dim // 2, embed_dim // 4)

        # 尺寸变化操作
        self.patch_embed1 = [UpEmbed(
            patch_size=3,
            in_chans=embed_dim,
            stride=1,
            padding=2,
            embed_dim=spec['DIM_EMBED'][0],
        ) for _ in range(self.task_no)]
        self.patch_embed1 = nn.ModuleList(self.patch_embed1)

        self.patch_embed2 = [UpEmbed(
            patch_size=3,
            in_chans=embed_dim // 2,
            stride=1,
            padding=2,
            embed_dim=spec['DIM_EMBED'][1],
        ) for _ in range(self.task_no)]
        self.patch_embed2 = nn.ModuleList(self.patch_embed2)
        # 多尺度融合信息的计算
        # 融合前的通道数变化信息
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
            cov1 = ConvBlock(1024, 144)
            self.fin_cov.append(cov1)
            pass
        pass

    def split_x(self, x, h, w):
        res = h*w
        x_list = []
        for i in range(self.task_no):
            _x = x[:, res*i:res*(i+1), :]
            x_list.append(_x)
        return x_list

    def forward(self, x_dict, inter_pred, back_fea, ori_feature):
        # 输入信息的产生
        x_list = []
        for task in self.all_tasks:
            _x = x_dict[task]
            _x = torch.cat([_x, inter_pred[task]], dim=1)
            _x = self.mix_proj[task](_x)
            x_list.append(_x)
            pass
        # print(ori_feature.shape)

        # 获得ecb40
        ecb40 = x_list
        ecb40 = self.Pself1(ecb40)
        ecb30 = back_fea[-2]
        temp_feature =[ecb40[i] + ecb30 for i in range(self.task_no)]
        # 获得dcb31
        dcb31 = temp_feature
        dcb31_ori = dcb31
        # 将dcb31的尺寸进行扩大
        dcb31 = [self.patch_embed1[i](dcb31[i]) for i in range(self.task_no)]
        # 获得ecb20
        ecb20 = back_fea[1]
        # 获得dcb21 由ecb30 和 ecb20 获得
        dcb21 = self.pag1(ecb20, self.compress1(ecb30))
        temp_feature = [ecb20 + dcb21 + dcb31[i] for i in range(self.task_no)]
        # 获得dcb22
        dcb22 = self.Pself2(temp_feature)
        dcb22_ori = dcb22
        # 对dcb22 进行尺寸的扩大
        dcb22 = [self.patch_embed2[i](dcb22[i]) for i in range(self.task_no)]
        # ecb10
        ecb10 = back_fea[0]
        # dcb11
        dcb11 = self.pag2(ecb10, self.compress2(ecb20))
        dcb12 = self.pag3(dcb11, self.compress3(dcb21))
        temp_feature = [dcb11 + ecb10 + dcb12 + dcb22[i] for i in range(self.task_no)]
        dcb13 = self.Pself3(temp_feature)
        x_dict = {}
        # 进行一个多尺度的融合，从而提升深度预测的能力
        #  另一尺度信息的获取
        oh, ow = self.p.spatial_dim[-1]
        ori_feature = rearrange(ori_feature, 'b (h w) c -> b c h w', h=oh, w=ow)
        for i, task in enumerate(self.all_tasks):
            h, w = dcb13[i].shape[2:]
            dcb31_ori[i] = self.update1[i](F.interpolate(dcb31_ori[i], size=(h,w),mode='bilinear',align_corners=False))
            dcb22_ori[i] = self.update2[i](F.interpolate(dcb22_ori[i], size=(h, w), mode='bilinear', align_corners=False))
            x_dict[task] = self.mix_res[task](dcb13[i] + dcb31_ori[i] + dcb22_ori[i])
            if i == 0:
                ori_feature = F.interpolate(ori_feature, size=(h, w), mode='bilinear', align_corners=False)
                pass
            task_feature = self.fin_cov[i](ori_feature)
            x_dict[task] = x_dict[task] + task_feature
        return x_dict
        pass
    pass


# 普通的尺寸变化模块
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


# 特殊的层次
class PselfAttention(nn.Module):
    def __init__(self, p, spec, i, dim_in, fea_no):
        super(PselfAttention, self).__init__()
        self.p = p
        self.attn = SelfAttention(spec=spec, index=i, fea_no=fea_no, dim_in=dim_in)
        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        self.stride_q = spec['STRIDE_Q'][i]
        self.task_no = fea_no
        dpr = [x.item() for x in torch.linspace(0, spec['DROP_PATH_RATE'][i], 1)]
        self.drop_path = DropPath(dpr[0]) \
            if dpr[0] > 0. else nn.Identity()
        mlp_ratio = spec['MLP_RATIO'][i]
        dim_mlp_hidden = int(dim_in * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_in,
            hidden_features=dim_mlp_hidden,
            act_layer=nn.GELU,
            drop=0.
        )
        self.part = nn.ModuleList([])
        self.mix = nn.ModuleList([])
        for i in range(fea_no):
            part = Part_feature(dim_in)
            self.part.append(part)
            mix = Mix_feature()
            self.mix.append(mix)
            pass
        pass

    def split_x(self, x, h, w):
        res = h*w
        x_list = []
        for i in range(self.task_no):
            _x = x[:, res*i:res*(i+1), :]
            x_list.append(_x)
        return x_list

    def forward(self, x_list):
        h, w = x_list[0].shape[2:]
        x_list_part = [self.part[i](_x) for i, _x in enumerate(x_list)]
        x_list = [rearrange(_x, 'b c h w -> b (h w) c') for _x in x_list]
        x = torch.cat(x_list, dim=1)
        res = x
        x = self.norm1(x)
        attn = self.attn(x, h, w)
        sh, sw = h // self.stride_q, w // self.stride_q
        attn_list = self.split_x(attn, sh, sw)
        attn_list = [rearrange(_it, 'b (h w) c -> b c h w', h=sh, w=sw) for _it in attn_list]
        attn_list = [F.interpolate(_it, size=(h, w), mode='bilinear', align_corners=False) for _it in attn_list]
        # todo 将另一个通道信息信息进行融合
        attn_list = [self.mix[i](attn_list[i], x_list_part[i])for i in range(self.task_no)]

        attn_list = [rearrange(_it, 'b c h w -> b (h w) c') for _it in attn_list]
        attn = torch.cat(attn_list, dim=1)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x_list = self.split_x(x, h, w)
        x_list = [rearrange(_it, 'b (h w) c -> b c h w', h=h, w=w) for _it in x_list]
        return x_list
        pass


#  不同于原始的自注意力机制，该自注意力要实现尺寸的变化
class SelfAttention(nn.Module):
    def __init__(self, spec, index, fea_no, dim_in):
        super(SelfAttention, self).__init__()

        self.stride_kv = spec['STRIDE_KV'][index]
        self.stride_q = spec['STRIDE_Q'][index]
        self.dim = dim_in
        self.num_heads = 2
        self.scale = self.dim ** -0.5
        self.fea_no = fea_no

        kernel_size_q = spec['KERNEL_Q'][index]
        padding_q = spec['PADDING_Q'][index]
        stride_q = spec['STRIDE_Q'][index]
        q_method = spec['Q_PROJ_METHOD'][index]
        kernel_size_kv = spec['KERNEL_KV'][index]
        padding_kv = spec['PADDING_KV'][index]
        stride_kv = spec['STRIDE_KV'][index]
        kv_method = spec['KV_PROJ_METHOD'][index]

        self.conv_proj_q = self._build_projection(
            dim_in, kernel_size_q, padding_q,
            stride_q, q_method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, kv_method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, kv_method
        )

        self.conv_proj_k1 = self._build_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, 'avg1'
        )

        self.conv_proj_v1 = self._build_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, 'avg1'
        )

        self.proj_q = nn.Linear(dim_in, self.dim, bias=True)
        self.proj_k = nn.Linear(dim_in, self.dim, bias=True)
        self.proj_v = nn.Linear(dim_in, self.dim, bias=True)

        self.attn_drop = nn.Dropout(0.15)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(0.)
        pass

    def _build_projection(self,
                          dim_in,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = [nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', BATCHNORM(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ])) for _ in range(self.fea_no)]
            proj = nn.ModuleList(proj)
        elif method == 'avg':
            proj = [nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ])) for _ in range(self.fea_no)]
            proj = nn.ModuleList(proj)
            pass
        elif method == 'avg1':
            proj = [nn.Sequential(OrderedDict([
                ('avg', nn.MaxPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ])) for _ in range(self.fea_no)]
            proj = nn.ModuleList(proj)
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    # 将向量重新变化为卷积的形式
    def split_x(self, x, h, w):
        res = h*w
        x_list = []
        for i in range(self.fea_no):
            _x = rearrange(x[:, res*i:res*(i+1), :], 'b (h w) c -> b c h w', h=h, w=w)
            x_list.append(_x)
        return x_list

    # 关键 获得q、k、v
    def forward_conv(self, x, h, w):
        x_list = self.split_x(x, h, w)
        # 计算q、k、v
        q_list = [self.conv_proj_q[i](x_list[i]) for i in range(self.fea_no)]
        q = torch.cat(q_list, dim=1)
        k_list = [self.conv_proj_k[i](x_list[i]) for i in range(self.fea_no)]
        k = torch.cat(k_list, dim=1)
        v_list = [self.conv_proj_v[i](x_list[i]) for i in range(self.fea_no)]
        v = torch.cat(v_list, dim=1)
        return q, k, v
        pass

    def forward(self, x, h, w):
        q, k, v = self.forward_conv(x, h, w)
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        # 计算相关性
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        # 注意力增强操作
        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
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


# 尺寸翻倍操作
class UpEmbed(nn.Module):
    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 ):
        super().__init__()
        patch_size = patch_size
        self.patch_size = patch_size

        self.proj = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, padding=padding, stride=stride, bias=False, dilation=padding),
                    BATCHNORM(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, padding=padding, stride=stride, bias=False, dilation=padding),
                    BATCHNORM(embed_dim),
                    nn.ReLU(inplace=True)
                    )

    def forward(self, x):
        x = self.proj(x)
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






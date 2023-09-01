# By Li Chang Hao
import torch
import torch.nn as nn
from models.PagUnet.pag_unet_perdecoder import PagUnet_PreDecoder
import torch.nn.functional as F
# 线性插值
INTERPOLATE_MODE = 'bilinear'
# resnet做为主干网络的 Pag-Unet模型代码
class Resnet_Net(nn.Module):
    def __init__(self,  p, backbone, backbone_channels, heads):
        super(Resnet_Net, self).__init__()
        self.p = p
        self.backbone =backbone
        self.heads = heads
        self.multi_task_decoder = PagUnet_PreDecoder(p)
        self.tasks = p.TASKS.NAMES
        pass

    def forward(self, x):
        #  记录最终的图片的尺寸
        img_size = x.size()[-2:]
        # 获取了经过主干网络处理后的资源
        x, selected_fea = self.backbone(x)
        # 获得中间结果和最终结果
        task_features, inter_preds = self.multi_task_decoder(selected_fea)
        out = task_features
        for t in self.tasks:
            out[t] = F.interpolate(self.heads[t](task_features[t]), img_size, mode=INTERPOLATE_MODE)
        out['inter_preds'] = {t: F.interpolate(v, img_size, mode=INTERPOLATE_MODE) for t, v in inter_preds.items()}
        return out
        pass
    pass

#  heads
class MLPHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MLPHead, self).__init__()
        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.linear_pred(x)
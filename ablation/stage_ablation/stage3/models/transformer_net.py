import torch
import torch.nn as nn
from models.transformers.pag_unet import PagUnet
import torch.nn.functional as F
INTERPOLATE_MODE = 'bilinear'

class TransformerNet(nn.Module):
    def __init__(self, p, backbone, backbone_channels, heads):
        super(TransformerNet, self).__init__()
        self.tasks = p.TASKS.NAMES
        self.backbone = backbone
        self.multi_task_decoder = PagUnet(p)
        self.heads = heads
        pass

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}
        # Backbone
        x, selected_fea = self.backbone(x)
        # transformers decoder
        task_features, inter_preds = self.multi_task_decoder(selected_fea)
        # Generate predictions
        out = task_features
        for t in self.tasks:
            out[t] = F.interpolate(self.heads[t](task_features[t]), img_size, mode=INTERPOLATE_MODE)
        out['inter_preds'] = {t: F.interpolate(v, img_size, mode=INTERPOLATE_MODE) for t, v in inter_preds.items()}
        return out
        pass


class MLPHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MLPHead, self).__init__()
        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.linear_pred(x)

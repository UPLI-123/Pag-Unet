# DRN 模型的主干网络， 维持了分辨率，使用扩张卷积保证感受野
import torch.nn as nn
import models.resnet as resnet


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial  # 可以为一个函数指定一个参数再参与运算

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            pass

        # 获得resnet 的定义（除了FC 和 avgpool）
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu

        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.feature_dim = orig_resnet.feature_dim
        pass


    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
                    pass
                pass
            pass
        pass

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x_list = []
        x = self.layer1(x)
        x_list.append(x)
        x = self.layer2(x)
        x_list.append(x)
        x = self.layer3(x)
        x_list.append(x)
        x = self.layer4(x)
        x_list.append(x)
        return x, x_list

    # 获得每一层的倒数第二层
    def forward_stage(self, x, stage):
        assert (stage in ['conv', 'layer1', 'layer2', 'layer3', 'layer4', 'layer1_without_conv'])

        if stage == 'conv':
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            return x

        elif stage == 'layer1':
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            return x

        elif stage == 'layer1_without_conv':
            x = self.layer1(x)
            return x

        else:  # Stage 2, 3 or 4
            layer = getattr(self, stage)
            return layer(x)

# 使用resnent 调用接口
def resnet_dilated(basenet, pretrained=True, dilate_scale=8):
    r"""Dilated Residual Network models from `"Dilated Residual Networks" <https://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_Dilated_Residual_Networks_CVPR_2017_paper.pdf>`_

    Args:
        basenet (str): The type of ResNet.
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        dilate_scale ({8, 16}, default=8): The type of dilating process.
    """
    return ResnetDilated(resnet.__dict__[basenet](pretrained=pretrained), dilate_scale=dilate_scale)

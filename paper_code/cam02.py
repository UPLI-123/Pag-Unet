import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import argparse
import torch
from utils.utils import mkdir_if_missing
from utils.common_config import get_model
from utils.config import create_config
import numpy as np
import matplotlib.pyplot as plt
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


# 配置信息的设置
parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--config_exp', default='./configs/nyud/nyud_vitLp16.yml',
                    help='Config file for the experiment')
parser.add_argument('--image_path', default='./image/cam.jpg',
                    help='Image path which has to be parsed')
parser.add_argument('--ckp_path', default='./cpk/checkpoint.pth.tar',
                    help='Config file for the experiment')
parser.add_argument('--save_dir', default="./output",
                    help='Save output image')
args = parser.parse_args()

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        # print(weight_softmax[idx,:,:,:].unsqueeze(0).shape)
        temp = weight_softmax[idx, :, :, :].reshape((512))
        temp = temp.cpu().numpy()
        print(temp)
        feature = feature_conv.reshape((nc, h*w)).cpu().numpy()
        print(feature)
        cam = temp.dot(feature)
        # cam = torch.mm(weight_softmax[idx,:,:,:].reshape((1,144)),feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

class DirectResize:
    """Resize samples so that the max dimension is the same as the giving one. The aspect ratio is kept.
    """

    def __init__(self, size):
        self.size = size

        self.mode = {
            'image': cv2.INTER_LINEAR
        }


    def resize(self, key, ori):
        new = cv2.resize(ori, self.size[::-1], interpolation=self.mode[key])

        return new

    def __call__(self, sample):
        for key, val in sample.items():
            if key == 'meta':
                continue
            sample[key] = self.resize(key, val)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_infer_transforms(p):
    from data import transforms
    import torchvision
    # dims = (3, 512, 512)
    dims = (3, 448, 576)
    valid_transforms = torchvision.transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.MaxResize(max_dim=512),
        DirectResize(dims[-2:]),
        # transforms.PadImage(size=dims[-2:]),
        transforms.ToTensor(),
    ])
    return valid_transforms

# 模型参数获取
def initialize_model(p, checkpoint_path):
    ckp = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckp['model']
    n_state_dict = {}
    for k, v in state_dict.items():
        print(k)
        n_state_dict[k] = v
    model = get_model(p)
    model.load_state_dict(n_state_dict)
    model = model.cuda()
    print("model initialized..")
    return model


# 该方法不经过梯度的计算
@torch.no_grad()
def infer(task):
    p = create_config(args.config_exp, {'run_mode': 'infer'})
    #  对准备输入的图片进行预处理
    checkpoint_path = args.ckp_path
    img = Image.open(args.image_path).convert('RGB')
    rgb_img = np.float32(img) / 255
    ori_size = img.size  # (w, h)
    img = np.asarray(img, dtype=np.float32)
    img = {'image': img}
    valid_transforms = get_infer_transforms(p)
    inp = valid_transforms(img)
    inp = inp['image']
    img = np.array(inp.permute(1, 2, 0))
    inp.requires_grad = False
    inp = inp.unsqueeze(0).cuda()
    # 加载模型 进行评估
    model = initialize_model(p, checkpoint_path)
    # print(model)
    model.multi_task_decoder.pre_decoder.preliminary_decoder[task].register_forward_hook(forward_hook)
    model.eval()
    with torch.no_grad():
        out = model(inp)
        pass
    # print(type(out[task]))
    normalized_masks = torch.nn.functional.softmax(out['inter_preds'][task], dim=1).cpu()
    print(normalized_masks.shape)
    mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    mask_max = np.max(mask)
    ids = []
    # 获取类别所对应的权重
    # weights = model._modules.get('heads').weight.data[cls, :]
    # weights =model.heads[task]
    # weight_softmax = model._modules.get('heads')[task].linear_pred.weight.data
    print(model)
    weight_softmax = model.multi_task_decoder.pre_decoder.intermediate_head[task].weight.data
    print(weight_softmax.shape)
    print(featute_map[0].shape)
    CAMs = returnCAM(featute_map[0], weight_softmax, ids)
    # print(CAMs)
    # render the CAM and output
    # img = cv2.imread('test.jpg')
    # height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], ori_size), cv2.COLORMAP_JET)
    img = cv2.imread(r'image\cam01.jpg')
    # heatmap = heatmap * 0.5 + img * 0.3
    # cv2.imwrite('CAM.jpg', result)
    name = "CAM01_"+task+".jpg"
    cv2.imwrite(name, heatmap)

    pass

featute_map = []
def forward_hook(module, inp, outp):
    featute_map.append(outp)
    pass





if __name__ == '__main__':
    # tasks = ['semseg', 'normals', 'sal', 'edge', 'human_parts']
    tasks = ['semseg']
    # 对每个任务进行可视化
    for task in tasks:
        infer(task)
        pass
    pass

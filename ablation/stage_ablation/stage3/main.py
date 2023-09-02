import argparse
import torch
import cv2
import random
import numpy as np
from utils.config import create_config
from utils.utils import mkdir_if_missing
from utils.common_config import get_transformations,get_test_dataloader,get_test_dataset,get_train_dataloader,get_train_dataset,get_model,\
    get_criterion, get_optimizer
from evaluation.evaluate_utils import PerformanceMeter
import os
import torch.nn as nn 
from utils.train_utils import train_phase

parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_exp', help='Config file for the experiment',default=r'/E22201107/Code/stage3/configs/pascal/pascal_vitLp16.yml')
# 随机参数
parser.add_argument('--run_mode',
                    help='Config file for the experiment', default='train')
parser.add_argument('--trained_model', default=None,
                    help='Config file for the experiment')
args = parser.parse_args()

# 判断是否可以放到GPU上运行
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 设置随机初始化种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    pass


def main():
    set_random_seed(0)
    params = {'run_mode': args.run_mode}
    p = create_config(args.config_exp, params)
    # print(p)
    # 1. 准备数据集
    train_transforms, val_transforms = get_transformations(p)
    if args.run_mode != 'infer':
        train_dataset = get_train_dataset(p, train_transforms)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        train_dataloader = get_train_dataloader(p, train_dataset)
    test_dataset = get_test_dataset(p, val_transforms)
    test_dataloader = get_test_dataloader(p, test_dataset)
    # 2. 准备模型
    model = get_model(p).cuda()
    # model = model.cuda()
    if torch.cuda.device_count() >1:
        print(torch.cuda.device_count())
        model = nn.DataParallel(model).cuda()
    # print(model)
    # 3. 准备评价指标
    criterion = get_criterion(p).cuda()
    # 4. 准备优化器
    scheduler, optimizer = get_optimizer(p, model)
    # 5.准备评价指标
    performance_meter = PerformanceMeter(p, p.TASKS.NAMES)

    # 检查点
    if os.path.exists(p['checkpoint']) or args.run_mode == 'infer':
        if args.trained_model != None:
            checkpoint_path = args.trained_model
        else:
            checkpoint_path = p['checkpoint']
        if args.local_rank == 0:
            print('Use checkpoint {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1  # epoch count is not used
        iter_count = checkpoint['iter_count']  # already + 1 when saving
    else:
        start_epoch = 0
        iter_count = 0
        pass

    # 准备训练
    # Train loop
    if args.run_mode != 'infer':
        for epoch in range(start_epoch, p['epochs']):
            end_signal, iter_count = train_phase(p, args, train_dataloader, test_dataloader, model, criterion, optimizer, scheduler, epoch, iter_count,device)
            if end_signal:
                break
                pass
            pass
    pass


if __name__ == '__main__':
    main()




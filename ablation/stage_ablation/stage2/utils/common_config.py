from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil
import torch


# 对数据进行的处理
def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import transforms
    import torchvision

    # Training transformations
    if p['train_db_name'] == 'NYUD' or p['train_db_name'] == 'PASCALContext':
        train_transforms = torchvision.transforms.Compose([ # from ATRC
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=p.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        # Testing
        valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TEST.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms

    else:
        return None, None

    pass


def get_train_dataset(p, transforms=None):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['train'], transform=transforms,
                                 retname=True,
                                 do_semseg='semseg' in p.TASKS.NAMES,
                                 do_edge='edge' in p.TASKS.NAMES,
                                 do_normals='normals' in p.TASKS.NAMES,
                                 do_sal='sal' in p.TASKS.NAMES,
                                 do_human_parts='human_parts' in p.TASKS.NAMES,
                                 overfit=False)

    if db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='train', transform=transforms,
                           do_edge='edge' in p.TASKS.NAMES,
                           do_semseg='semseg' in p.TASKS.NAMES,
                           do_normals='normals' in p.TASKS.NAMES,
                           do_depth='depth' in p.TASKS.NAMES, overfit=False)

    return database


def get_train_dataloader(p, dataset):
    """ Return the train dataloader """
    collate = collate_mil
    trainloader = DataLoader(dataset, batch_size=p['trBatch'], drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate, pin_memory=True)
    return trainloader


def get_test_dataset(p, transforms=None):
    """ Return the test dataset """

    db_name = p['val_db_name']
    print('Preparing test dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['val'], transform=transforms,
                                 retname=True,
                                 do_semseg='semseg' in p.TASKS.NAMES,
                                 do_edge='edge' in p.TASKS.NAMES,
                                 do_normals='normals' in p.TASKS.NAMES,
                                 do_sal='sal' in p.TASKS.NAMES,
                                 do_human_parts='human_parts' in p.TASKS.NAMES,
                                 overfit=False)

    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='val', transform=transforms,
                           do_edge='edge' in p.TASKS.NAMES,
                           do_semseg='semseg' in p.TASKS.NAMES,
                           do_normals='normals' in p.TASKS.NAMES,
                           do_depth='depth' in p.TASKS.NAMES)

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_test_dataloader(p, dataset):
    """ Return the validation dataloader """
    collate = collate_mil
    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'], pin_memory=True, collate_fn=collate)
    return testloader


def get_head(p, task):
    '''
    返回 解码器的头
    :param p:
    :param feat_channels:
    :param task:
    :return:
    '''
    if p['head'] == 'mlp':
        from models.transformer_net import MLPHead
        return MLPHead(576, p.TASKS.NUM_OUTPUT[task])
        pass

    pass


def get_backbone(p):
    """ Return the backbone """

    if p['backbone'] == 'vitL':
        from models.transformers.vit import vit_large_patch16_384
        backbone = vit_large_patch16_384(pretrained=True, drop_path_rate=0.15, img_size=p.TRAIN.SCALE)
        backbone_channels = [1024 for _ in range(4)]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//16, p.TRAIN.SCALE[1]//16] for _ in range(4)]
        p.final_embed_dim = p.embed_dim + p.PRED_OUT_NUM_CONSTANT
    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_model(p):
    '''返回模型参数'''
    # 获得主干网络
    backbone, backbone_channels = get_backbone(p)
    #  设置自己使用好的网络
    if p['model'] == 'TransformerNet':
        from models.transformer_net import TransformerNet
        # 最终的通道数
        heads = torch.nn.ModuleDict({task: get_head(p, task) for task in p.TASKS.NAMES})
        model = TransformerNet(p, backbone, backbone_channels, heads)
        return model
        pass
    pass
# 每个任务 损失函数的计算
def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(ignore_index=p.ignore_index)

    elif task == 'normals':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=True, ignore_index=p.ignore_index)

    elif task == 'sal':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index)

    elif task == 'depth':
        from losses.loss_functions import L1Loss
        criterion = L1Loss()

    else:
        criterion = None

    return criterion


def get_criterion(p):
    from losses.loss_schemes import MultiTaskLoss
    loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
    loss_weights = p['loss_kwargs']['loss_weights']
    return MultiTaskLoss(p, p.TASKS.NAMES, loss_ft, loss_weights)


def get_optimizer(p, model):
    """ Return optimizer for a given model and setup """

    print('Optimizer uses a single parameter group - (Default)')
    params = model.parameters()

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    # get scheduler
    if p.scheduler == 'poly':
        from utils.train_utils import PolynomialLR
        scheduler = PolynomialLR(optimizer, p.max_iter, gamma=0.9, min_lr=0)
    elif p.scheduler == 'step':
        scheduler = torch.optim.MultiStepLR(optimizer, milestones=p.scheduler_kwargs.milestones,
                                            gamma=p.scheduler_kwargs.lr_decay_rate)

    return scheduler, optimizer

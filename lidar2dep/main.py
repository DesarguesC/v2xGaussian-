"""
    CompletionFormer
    ======================================================================

    main script for training and testing.
"""


from config import args as args_config
import random, os, json, torch
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
torch.autograd.set_detect_anomaly(True)

import utility
from model.completionformer import CompletionFormer
from summary.cfsummary import CompletionFormerSummary
from metric.cfmetric import CompletionFormerMetric
from data import get as get_data
from loss.l1l2loss import L1L2Loss

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Minimize randomness
def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume

    return new_args

from data.process import *


def get_CompletionFormer(args):
    net = CompletionFormer(args)
    net.cuda()
    checkpoint = torch.load(args.pretrain)
    key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)
    if key_u:
        print('Unexpected keys :')
        print(key_u)

    if key_m:
        print('Missing keys :')
        print(key_m)
        raise KeyError
    print('Checkpoint loaded from {}!'.format(args.pretrain))
    net = nn.DataParallel(net)
    net.eval()
    return net



if __name__ == '__main__':
    opt = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(opt)):
        print(key, ':',  getattr(opt, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    init_seed()
    """
        --pretrain:     weights path
        --rgb_file_path:        where to locate rgb image file
        --pcd_file_path:        where to lcoate v2x-i view
        --camera_file_path:     where to put the camera
    """


    """
        {'rgb': rgb, 'depth': depth, 'K': torch.Tensor(K)}
        rgb:        torch.Tensor
        depth:      torch.Tensor
        K:          torch.Tensor
    """
    I_dict = pre_read(opt.rgb_file_path. args.pcd_file_path, opt.camra_file_path)
    net = get_CompletionFormer(opt)
    rgb, depth, K = I_dict['rgb'], I_dict['dep'], I_dict['K']
    sample = {
        'rgb': rgb,
        'dep': depth
    }
    out = net(sample)
    # use: pdb
    # TODO: check data format

    


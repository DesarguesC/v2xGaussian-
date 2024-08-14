"""
    CompletionFormer
    ======================================================================

    main script for training and testing.
"""
import pdb
from einops import repeat, rearrange
from config import args as args_config
import random, os, json, torch
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

from torch import nn
from torch.nn.functional import interpolate as Inter
torch.autograd.set_detect_anomaly(True)
import utility
from model.completionformer import CompletionFormer

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


def get_new_size(shape, threshold=1e3-200):
    assert len(shape) >= 3, shape
    a, b = shape[2 if len(shape) == 4 else 1:]
    s = 1.
    while a > threshold or b > threshold:
        a /= s
        b /= s
        s += 0.5

    a = int(a/64+0.5) * 64
    b = int(b/64+0.5) * 64

    return (a, b)


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




def main():
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
        --intrinsic_path:       intrinsic file JSON
        --extrinsic_path:       extrinsic file JSON
    """


    """
        {'rgb': rgb, 'depth': depth, 'K': torch.Tensor(K)}
        rgb:        torch.Tensor
        depth:      torch.Tensor
        K:          torch.Tensor
    """

    # from config import parser as opt
    I_dict = pre_read(opt.rgb_file_path, opt.pcd_file_path, opt.intrinsic_path, opt.extrinsic_path)
    net = get_CompletionFormer(opt)
    rgb, depth, K = I_dict['rgb'], I_dict['dep'], I_dict['K']
    # K: intrinsic matrix -> torch.Tensor[3 3]
    if len(rgb.shape) <= 3:
        rgb = rgb.unsqueeze(0)
    if len(depth.shape) <= 3:
        depth = depth.unsqueeze(0)

    assert len(rgb.shape) == 4 and len(depth.shape) == 4, f'rgb.shape = {rgb.shape}, dep.shape = {depth.shape}'
    rgb_size = get_new_size(rgb.shape)
    dep_size = get_new_size(depth.shape)
    rgb = Inter(torch.tensor(rgb, dtype=torch.float32), size=rgb_size, mode="bilinear")
    depth = Inter(torch.tensor(depth, dtype=torch.float32), size=dep_size, mode="bilinear")

    sample = {
        'rgb': rgb.cuda(),     # torch.Tensor[1, 3, H, W]
        'dep': depth.cuda()    # torch.Tensor[1, 1, H, W]
    }

    out = net(sample)
    # use: pdb
    # TODO: check data format

    print(out)
    pred = out['pred'].squeeze() # [1 1 H W]
    pred = pred.detach().cpu().numpy().astype(np.uint8)
    M, m = np.max(pred), np.min(pred)
    pred = (pred - m) / (M - m) * 255.
    colored_pred = cv2.applyColorMap(pred.astype(np.uint8), cv2.COLORMAP_JET)
    print(pred)
    cv2.imwrite('../data/depth-tmp/test++.jpg', cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))

    pred_init = out['pred_init'].squeeze()
    pred_init = pred_init.detach().cpu().numpy().astype(np.uint8)
    M, m = np.max(pred_init), np.min(pred_init)
    pred_init = (pred_init - m) / (M - m) * 255.
    colored_init = cv2.applyColorMap(pred_init.astype(np.uint8), cv2.COLORMAP_JET)
    print(colored_init)
    cv2.imwrite('../data/depth-tmp/init.jpg', cv2.cvtColor(colored_init, cv2.COLOR_RGB2BGR))




if __name__ == '__main__':
    main()


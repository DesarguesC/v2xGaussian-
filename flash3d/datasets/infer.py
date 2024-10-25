import os, pdb
import random
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T

from PIL import Image
from typing import Optional
from pathlib import Path
from flash3d.datasets.data import pil_loader

class InferenceV2X(data.Dataset):
    def __init__(self, dair_info, num_scales: int = 1, view_type: str = 'inf'):
        self.is_train = True
        self.num_scales = num_scales
        self.type = view_type
        # dair_info: drgs_util.scene.dataset_renders.Dair_v2x_Info[class]
        self.img = getattr(dair_info, f'{view_type}_rgb') # np.ndarray
        self.image_size = self.img.shape[:-1]
        # only RGB image here, add depth prior after
        self.pcd = getattr(dair_info, f'{view_type}_pcd') # basic point cloud type
        self.depth = getattr(dair_info, f'{view_type}_depth')
        self.cam2world = np.linalg.inv(getattr(dair_info, f'world2cam_{view_type}'))
        """
            dict{'fg':..., 'bg':..., 'panoptic':...}
        """
        self.K = getattr(dair_info, f'{view_type}_cam_K')
        """
            dict{'height':..., 'weight':..., 'cam_K': np.ndarray}
        """
        self.intrinsics = getattr(dair_info, f'normalization_{view_type}')["radius"]
        # 处理好的相机内参
        """
        ORIGINAL - calibs
        {
            "K": K,
            "K_raw": K_raw,
            "T_l": T_l,
            "T_r": T_r,
            "P_v2cl": P_v2cl,
            "P_v2cr": P_v2cr,
            "crop": box
        }
        """
        frame_idxs = [0, 1, 2] # as written in flash3d/configs/model/gaussian.yaml
        self.frame_idxs = frame_idxs
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.pad_border_aug = 0 # DAIR-V2X is KITTI-like dataset


    def __len__(self):
        return 1

    def __getitem__(self, item):

        inputs = {}
        # only single
        try_flag = True # DEBUG
        while try_flag:
            try_flag = False
            try:
                # self.num_scales = 1
                for scale in range(self.num_scales):
                    K = self.K
                    K_tgt = K.copy()
                    K_src = K.copy()

                    K_tgt[0, :] *= self.image_size[1] // (2 ** scale)
                    K_tgt[1, :] *= self.image_size[0] // (2 ** scale)

                    K_src[0, :] *= self.image_size[1] // (2 ** scale)
                    K_src[1, :] *= self.image_size[0] // (2 ** scale)
                    # principal points change if we add padding
                    K_src[0, 2] += self.pad_border_aug // (2 ** scale)
                    K_src[1, 2] += self.pad_border_aug // (2 ** scale)

                    inv_K_src = np.linalg.pinv(K_src)

                    inputs[("K_tgt", scale)] = torch.from_numpy(K_tgt)[..., :3, :3]
                    inputs[("K_src", scale)] = torch.from_numpy(K_src)[..., :3, :3]
                    inputs[("inv_K_src", scale)] = torch.from_numpy(inv_K_src)[..., :3, :3]


                inputs[("color", 0, -1)] = Image.fromarray(self.img)
                # only single frame
                inputs[("depth_gt", 0, 0)] = self.depth
                inputs[("T_c2w", 0)] = self.cam2world

            except Exception as err:
                print(f'err = {err}')
                try_flag = True
                pdb.set_trace()


        return inputs



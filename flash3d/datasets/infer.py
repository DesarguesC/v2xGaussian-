import os, pdb
import random
import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image
from typing import Optional
from pathlib import Path
from flash3d.datasets.data import pil_loader

class InferenceV2X:
    def __init__(self, split, cfg, dair_info, num_scales: int = 1, view_type: str = 'inf'):
        self.is_train = True
        self.num_scales = num_scales
        self.cfg = cfg
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

        self.pad_border_fn = T.Pad((self.cfg.dataset.pad_border_aug, self.cfg.dataset.pad_border_aug))
        self.interp = Image.LANCZOS
        self.to_tensor = T.ToTensor()
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
        if cfg.model.gaussian_rendering:
            frame_idxs = [0] + cfg.model.gauss_novel_frames
            if cfg.dataset.stereo:
                if split == "train":
                    stereo_frames = []
                    for frame_id in frame_idxs:
                        stereo_frames += [f"s{frame_id}"]
                    frame_idxs += stereo_frames
                else:
                    frame_idxs = [0, "s0"]
        else:
            # SfMLearner frames, eg. [0, -1, 1]
            frame_idxs = cfg.model.frame_ids.copy()

        self.frame_idxs = frame_idxs
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
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            new_size = (self.image_size[0] // s, self.image_size[1] // s)
            self.resize[i] = T.Resize(new_size, interpolation=self.interp)

        self.resize_depth = T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)


    def __len__(self):
        return 1

    def preprocess(self, inputs, color_aug, device='cuda'):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)]).to(device)


        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                if self.cfg.dataset.pad_border_aug != 0:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(self.pad_border_fn(color_aug(f))).to(torch.float32).to(device)
                else:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f)).to(torch.float32).to(device)

        return inputs

    def getInputs(self, device='cuda'):
        print('getting inference item...')
        cfg = self.cfg
        inputs = {}

        do_color_aug = cfg.dataset.color_aug and self.is_train and random.random() > 0.5
        # do_flip = cfg.dataset.flip_left_right and self.is_train and random.random() > 0.5

        frame_idxs = list(self.frame_idxs).copy()

        # for f_id in frame_idxs:
        #     if type(f_id) == str and f_id[0] == "s":  # stereo frame
        #         the_side = stereo_flip[side]
        #         i = int(f_id[1:])
        #     else:
        #         the_side = side
        #         i = f_id
        #     inputs[("color", f_id, -1)] = self.get_color(folder, frame_index + i, the_side, do_flip)
        #
        # inputs[("frame_id", 0)] = \
        #     f"{os.path.split(folder)[1]}+{side}+{frame_index:06d}"

        # only single
        try_flag = True # DEBUG
        while try_flag:
            try_flag = False
            try:
                # self.num_scales = 1
                for scale in range(self.num_scales):
                    # pdb.set_trace()
                    K = self.K['cam_K']
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

                    inputs[("K_tgt", scale)] = torch.from_numpy(K_tgt)[..., :3, :3].to(device)
                    inputs[("K_src", scale)] = torch.from_numpy(K_src)[..., :3, :3].to(device)
                    inputs[("inv_K_src", scale)] = torch.from_numpy(inv_K_src)[..., :3, :3].to(device)

                if do_color_aug:
                    raise NotImplementedError
                    color_aug = random_color_jitter(self.brightness, self.contrast, self.saturation, self.hue)
                else:
                    color_aug = (lambda x: x)

                # only single frame
                inputs[("color", 0, -1)] = Image.fromarray(self.img)  # need tensor ?
                inputs[("depth_gt", 0, 0)] = self.depth  # need tensor ?
                inputs[("T_c2w", 0)] = self.cam2world  # need tensor ?

                inputs = self.preprocess(inputs, color_aug)

                # self.to_tensor(color_aug_fn(self.pad_border_fn(img_scale)))





            except Exception as err:
                pdb.set_trace()
                print(f'err = {err}')
                try_flag = True


        return inputs

    def __getitem__(self, item):
        return self.getInputs()

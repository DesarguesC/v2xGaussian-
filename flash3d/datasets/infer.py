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
        self.anti_type = 'veh' if view_type=='inf' else 'inf'
        # pdb.set_trace()
        # dair_info: drgs_util.scene.dataset_renders.Dair_v2x_Info[class]
        self.img = {
            '0': getattr(dair_info, f'{view_type}_rgb'),
            's0': getattr(dair_info, f'{self.anti_type}_rgb')
        } # np.ndarray
        self.image_size = self.img['0'].shape[:-1]
        # only RGB image here, add depth prior after
        self.pcd = {
            '0': getattr(dair_info, f'{view_type}_pcd'),
            's0': getattr(dair_info, f'{self.anti_type}_pcd')
        } # basic point cloud type
        self.depth = {
            '0': getattr(dair_info, f'{view_type}_depth'),
            's0': getattr(dair_info, f'{self.anti_type}_depth')
        }
        self.cam2world = {
            '0': np.linalg.inv(getattr(dair_info, f'world2cam_{view_type}')),
            's0': np.linalg.inv(getattr(dair_info, f'world2cam_{self.anti_type}'))
        }
        """
            dict{'fg':..., 'bg':..., 'panoptic':...}
        """
        # pdb.set_trace()
        self.K = {
            '0': getattr(dair_info, f'{view_type}_cam_K'),
            's0': getattr(dair_info, f'{self.anti_type}_cam_K')
        } # check the shape of 'cam_K'
        """
            dict{'height':..., 'weight':..., 'cam_K': np.ndarray}
        """
        self.intrinsics = {
            '0': getattr(dair_info, f'normalization_{view_type}')["radius"],
            's0': getattr(dair_info, f'normalization_{self.anti_type}')["radius"]
        }

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

        # pdb.set_trace()
        self.frame_idxs = frame_idxs
        # frame_idxs = [0, 1, 2] # as written in flash3d/configs/model/gaussian.yaml
        # self.frame_idxs = frame_idxs

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
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        # pdb.set_trace()
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f).to(device)
                if self.cfg.dataset.pad_border_aug != 0:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(self.pad_border_fn(color_aug(f))).to(torch.float32).to(device)
                else:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f)).to(torch.float32).to(device)

                if len(inputs[(n + "_aug", im, i)].shape) < 4 and inputs[(n + "_aug", im, i)].shape[0] == 3:
                    inputs[(n + "_aug", im, i)] = inputs[(n + "_aug", im, i)][None,:,:,:]
                    inputs[(n, im, i)] = inputs[(n, im, i)][None, :, :, :]
                    print(inputs[(n + "_aug", im, i)].shape)
                print(f'{inputs[(n + "_aug", im, i)].shape}, {inputs[(n, im, i)].shape}')
                # if len(inputs[(n + "_aug", im, i)].shape) < 3:
                #     pdb.set_trace()
                #     inputs[(n + "_aug", im, i)] = inputs[(n + "_aug", im, i)][None,:,:] # will be turned to float64 again?
                #     inputs[(n + "_aug", im, i)] = inputs[(n + "_aug", im, i)].to(torch.float32)

        return inputs

    def getInputs(self, device='cuda'):
        print('getting inference item...')
        cfg = self.cfg
        inputs = {}

        do_color_aug = cfg.dataset.color_aug and self.is_train and random.random() > 0.5
        # do_flip = cfg.dataset.flip_left_right and self.is_train and random.random() > 0.5

        frame_idxs = list(self.frame_idxs).copy() # target_frame_ids ?

        # TODO: inputs[("frame_id", 0)] = f"{os.path.split(folder)[1]}+{side}+{frame_index:06d}" 【路径】

        # only single
        try_flag = True # DEBUG
        while try_flag:
            try_flag = False
            try:
                # self.num_scales = 1
                for scale in range(self.num_scales):
                    # pdb.set_trace()
                    K = self.K['0']['cam_K']
                    K_tgt = K.copy()
                    K_src = K.copy()

                    K_tgt[0, :] *= self.image_size[1] // (2 ** scale)
                    K_tgt[1, :] *= self.image_size[0] // (2 ** scale)

                    K_src[0, :] *= self.image_size[1] // (2 ** scale)
                    K_src[1, :] *= self.image_size[0] // (2 ** scale)
                    # principal points change if we add padding
                    K_src[0, 2] += self.pad_border_aug // (2 ** scale)
                    K_src[1, 2] += self.pad_border_aug // (2 ** scale)

                    inv_K_src = np.linalg.pinv(K_src) # Shape[3,3]

                    inputs[("K_tgt", scale)] = torch.from_numpy(K_tgt)[None,:,:].to(device)
                    inputs[("K_src", scale)] = torch.from_numpy(K_src)[None,:,:].to(device)
                    inputs[("inv_K_src", scale)] = torch.from_numpy(inv_K_src)[None,:,:].to(device)

                if do_color_aug:
                    raise NotImplementedError
                    color_aug = random_color_jitter(self.brightness, self.contrast, self.saturation, self.hue)
                else:
                    color_aug = (lambda x: x)

                # only single frame
                for f_id in frame_idxs:
                    inputs[("color", f_id, -1)] = Image.fromarray(self.img[str(f_id)])
                    inputs[("depth_gt", f_id, 0)] = self.depth[str(f_id)]
                    # pdb.set_trace()
                    inputs["depth_sparse", f_id] = np.asarray(self.pcd[str(f_id)].points)[None,:,:] # check shape & type
                    # just use point cloud coordinates
                    inputs[("T_c2w", f_id)] = torch.tensor(self.cam2world[str(f_id)][None,:,:], dtype=torch.float32).to(device)

                inputs = self.preprocess(inputs, color_aug)

                for i in frame_idxs:
                    del inputs[("color", i, -1)]
                    del inputs[("color_aug", i, -1)]

                    # self.to_tensor(color_aug_fn(self.pad_border_fn(img_scale)))

            except Exception as err:
                pdb.set_trace()
                print(f'err = {err}')
                try_flag = True


        return inputs

    def __getitem__(self, item):
        return self.getInputs()

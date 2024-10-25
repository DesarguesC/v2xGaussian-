import os
import random
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T

from PIL import Image
from typing import Optional
from pathlib import Path
from flash3d.datasets.data import pil_loader

class inference:
    def __init__(self, dair_info, type: str = 'inf'):
        # dair_info: drgs_util.scene.dataset_renders.Dair_v2x_Info[class]
        self.img = getattr(dair_info, f'{type}_rgb') # np.ndarray
        # only RGB image here, add depth prior after
        self.pcd = getattr(dair_info, f'{type}_pcd') # basic point cloud type
        self.depth = getattr(dair_info, f'{type}_depth')
        """
            dict{'fg':..., 'bg':..., 'panoptic':...}
        """
        self.cam_K = getattr(dair_info, f'{type}_cam_K')
        """
            dict{'height':..., 'weight':..., 'cam_K': np.ndarray}
        """



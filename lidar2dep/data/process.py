# reference: https://github.com/youmi-zym/CompletionFormer/blob/main/src/data/kittidc.py
import os, json, random
import open3d as o3d
import numpy as np
from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from .lidar import sample_lidar_lines

# Reference : https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

# pcd is namely the sparse depth ?
def pre_read(rgb_file_path, pcd_file_path, camera_file_path, lidar_lines=64, return_tensor=True):
    # L109 -> L247 -> L91, L285
    keep_ratio = (lidar_lines / 64.0)
    assert keep_ratio >=0 and keep_ratio <= 1.0, keep_ratio
    if keep_ratio >= 0.9999:
            pass
    elif keep_ratio > 0:
        Km = np.eye(3)
        Km[0, 0] = K[0]
        Km[1, 1] = K[1]
        Km[0, 2] = K[2]
        Km[1, 2] = K[3]
        depth = sample_lidar_lines(depth[:, :, None], intrinsics=Km, keep_ratio=keep_ratio)[:, :, 0]
    else:
        depth = np.zeros_like(depth)


    # TODO: 找到rgb图片视角下的pcd渲染出的sparse depth
    rgb_image = Image.open(rgb_file_path)
    # width, height = rgb_image.size

    with open(camera_file_path, 'r') as f:
        camera_file = json.load(f)
    K = np.concatenate([np.array(camera_file['rotation']), np.array(camera_file['translation'])], axis=-1)
    
    pcd_file = o3d.io.read_point_cloud(pcd_file_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd_file)
    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(K) # To render the cooresponding view
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image)
    depth_image = (image * 255).astype(np.uint8) # np -> cv2
    vis.destroy_window()

    sampled_depth = sample_lidar_lines(depth_image, np.array(camera_file['rotation']))

    if not return_tensor:
        return {'rgb': rgb_image, 'depth': sampled_depth, 'K': torch.Tensor(K)}
    else:
        rgb = TF.to_tensor(rgb_image)
        rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        depth = TF.to_tensor(np.array(sampled_depth))
        return {'rgb': rgb, 'depth': depth, 'K': torch.Tensor(K)}






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

def create_former_input(
        rgb_np_img: np.ndarray,
        pcd_np_img:np.ndarray,
        camera_intrinsic: np.ndarray,
        keep_ratio: float = 1.
    ):
    assert camera_intrinsic.shape == (3,3), camera_intrinsic
    assert keep_ratio >= 0 and keep_ratio <= 1.0, keep_ratio
    assert len(camera_intrinsic.keys()) == 5, camera_intrinsic

    if np.max(pcd_np_img) <= 1.: pcd_np_img = (pcd_np_img * 255).astype(np.uint8)
    if np.max(rgb_np_img) <= 1.: rgb_np_img = (rgb_np_img * 255).astype(np.uint8)

    sampled_depth = sample_lidar_lines(
        depth_map=pcd_np_img, intrinsics=camera_intrinsic, keep_ratio=keep_ratio
    )

    return {
        'rgb': rgb_np_img,
        'dep': sampled_depth,
        'K_matrix': torch.Tensor(camera_intrinsic)
    }




# pcd is namely the sparse depth ?
def pre_read(rgb_file_path, pcd_file_path, intrinsic_path, extrinsic_path, lidar_lines=64, return_tensor=True):
    # L109 -> L247 -> L91, L285

    # TODO: read camera intrinsics
    with open(intrinsic_path) as f:
        intrinsics = json.load(f)
    cam_D, cam_K = intrinsics['cam_D'], intrinsics['cam_K']
    # cam_D seems useless
    K_dict = {
        'height': intrinsics['height'], 'width': intrinsics['width'],
        'fx': cam_K[0], 'fy': cam_K[4], 'cx': cam_K[2], 'cy': cam_K[5]
    }
    K_matrix = np.array(cam_K).reshape((3,3))

    # TODO: read camera extrinsics
    with open(extrinsic_path) as f:
        extrinsics = json.load(f)
    R, T = np.array(extrinsics['rotation']), np.array(extrinsics['translation'])
    A = np.zeros((4,4))
    A[0:3, 0:3] = R
    A[0:3, -1] = np.squeeze(T)
    A[-1, -1] = 1.

    keep_ratio = (lidar_lines / 64.0)
    assert keep_ratio >=0 and keep_ratio <= 1.0, keep_ratio
    # if keep_ratio >= 0.9999:
    #         pass
    # elif keep_ratio > 0:
    #     Km = np.eye(3)
    #     Km[0, 0] = K[0]
    #     Km[1, 1] = K[1]
    #     Km[0, 2] = K[2]
    #     Km[1, 2] = K[3]
    #     depth = sample_lidar_lines(depth[:, :, None], intrinsics=Km, keep_ratio=keep_ratio)[:, :, 0]
    # else:
    #     depth = np.zeros_like(depth)


    # TODO: 找到rgb图片视角下的pcd渲染出的sparse depth
    rgb_image = Image.open(rgb_file_path)
    # width, height = rgb_image.size

    pcd_file = o3d.io.read_point_cloud(pcd_file_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=K_dict['width'], height=K_dict['height'])
    vis.add_geometry(pcd_file)

    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.extrinsic = A # np.array [4,4]
    camera_parameters.intrinsic.set_intrinsics(**K_dict)


    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera_parameters) # To render the cooresponding view
    vis.poll_events()
    vis.update_renderer()

    pcd_img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    # TODO: check if this multiplication is need
    depth_image = (pcd_img * 255).astype(np.uint8)

    sampled_depth = sample_lidar_lines(
        depth_map = depth_image, intrinsics = K_matrix, keep_ratio=keep_ratio
    )

    if not return_tensor:
        return {'rgb': rgb_image, 'depth': sampled_depth, 'K': torch.Tensor(K_matrix)}
    else:
        rgb = TF.to_tensor(rgb_image)
        rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        depth = TF.to_tensor(np.array(sampled_depth))
        return {'rgb': rgb, 'dep': depth, 'K': torch.Tensor(K_matrix)}







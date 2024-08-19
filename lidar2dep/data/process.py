# reference: https://github.com/youmi-zym/CompletionFormer/blob/main/src/data/kittidc.py
import os, json, random, pdb, cv2
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


def fix_pcd(image, mask): # mask: fg-mask -> cars
    if mask is None: return image
    u = torch.tensor(image, dtype=torch.float32)
    u[torch.tensor(mask, dtype=torch.bool)] = 1.
    assert torch.sum(u*mask - mask).item() == 0, torch.sum(u*mask - mask)
    return np.array(u)


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
def pre_read(
        depth_path, rgb_file_path, pcd_file_path,
        intrinsic_path, extrinsic_path, fg_mask=None,
        extra_name='fg', lidar_lines=64, return_tensor=True
):
    # L109 -> L247 -> L91, L285
    # Note the 'depth_path' only related to saving directory
    # TODO: read camera intrinsics
    if fg_mask is not None: assert not isinstance(rgb_file_path, str)
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


    # TODO: 找到rgb图片视角下的pcd渲染出的sparse depth
    rgb_image = Image.open(rgb_file_path) if isinstance(rgb_file_path, str) else Image.fromarray(rgb_file_path)
    pcd_file = o3d.io.read_point_cloud(pcd_file_path)


    # Off Screen Rendering
    renderer = o3d.visualization.rendering.OffscreenRenderer(K_dict['width'], K_dict['height'])
    material = o3d.visualization.rendering.MaterialRecord()
    material.point_size = 5
    material.shader = 'unlit' # maintain original colors
    renderer.scene.add_geometry("point_cloud", pcd_file, material)

    renderer.setup_camera(K_matrix, A, K_dict['width'], K_dict['height'])
    pcd_img = np.asarray(renderer.render_to_depth_image())
    # Standard Nom First
    # depth_image = (depth_image - np.mean(depth_image)) / np.std(depth_image)

    # Max-Min Norm
    # TODO -> Change Here !
    if fg_mask is not None:
        try:
            fix_pcd(pcd_img, fg_mask)
        except Exception as err:
            print(f'err: {err}')
            pdb.set_trace()

    # 把mask部分的pcd点抹去，应该是把mask到的部分变白
    # pcd_img: [H W], fg_mask: [H W 3]
    M, m  = np.max(pcd_img), np.min(pcd_img)
    depth_image = (pcd_img - m) / (M - m) * 255.

    colored_depth = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_RAINBOW)
    cv2.imwrite(os.path.join(depth_path, f'projected_pcd-{extra_name}.jpg'), cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR))

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




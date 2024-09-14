#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, cv2, sys, torch, pdb
# ab_path = os.path.dirname(os.path.abspath(sys.argv[0]))
# sys.path.append(os.path.join(ab_path, 'utils'))
from pytorch3d.transforms.so3 import (
    so3_exp_map,
    so3_log_map,
    so3_relative_angle,
)
from pytorch3d.renderer.cameras import (
    SfMPerspectiveCameras,
)

from glob import glob
from PIL import Image
from typing import NamedTuple
from .colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, rotmat2qvec, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

from drgs_utils import ModelParams
from ..utils import *
from lidar2dep.dair import CooperativeData

import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from .gaussian_model import BasicPointCloud
import imageio
from datetime import datetime
from tqdm import tqdm



class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array = None
    depth_weight: np.array = None
    mask: np.array = None
    depthloss: float=1e5

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str



def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, pcd=None, resolution=4, train_idx=None, white_background=False):
    cam_infos = []
    model_zoe = None

    for idx, key in enumerate(sorted(cam_extrinsics,key=lambda x:cam_extrinsics[x].name)):
        
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path).resize((width//resolution, height//resolution))
        
        if white_background:
            ############################## borrow from blender ##################################
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1,0]) if white_background else np.array([0, 0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:4] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
            #######################################################################################
        
        
        depthmap, depth_weight = None, None
        depthloss = 1e8
        if pcd is not None and idx in train_idx:
            depthmap, depth_weight = np.zeros((height//resolution,width//resolution)), np.zeros((height//resolution,width//resolution))
            K = np.array([[focal_length_x, 0, width//resolution/2],[0,focal_length_y,height//resolution/2],[0,0,1]])
            cam_coord = np.matmul(K, np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3,1)) ### for coordinate definition, see getWorld2View2() function
            valid_idx = np.where(np.logical_and.reduce((cam_coord[2]>0, cam_coord[0]/cam_coord[2]>=0, cam_coord[0]/cam_coord[2]<=width//resolution-1, cam_coord[1]/cam_coord[2]>=0, cam_coord[1]/cam_coord[2]<=height//resolution-1)))[0]
            pts_depths = cam_coord[-1:, valid_idx]
            cam_coord = cam_coord[:2, valid_idx]/cam_coord[-1:, valid_idx]
            depthmap[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = pts_depths
            depth_weight[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = 1/pcd.errors[valid_idx] if pcd.errors is not None else 1
            depth_weight = depth_weight/depth_weight.max()

            if model_zoe is None:
                model_zoe = torch.hub.load("./ZoeDepth", "ZoeD_NK", source="local", pretrained=True).to('cuda')
            
            source_depth = model_zoe.infer_pil(image.convert("RGB"))
            target=depthmap.copy()
            
            target=((target != 0) * 255).astype(np.uint8)
            depthmap, depthloss = optimize_depth(source=source_depth, target=depthmap, mask=depthmap>0.0, depth_weight=depth_weight)
            
            import cv2
            from drgs_utils.gaussian_renderer import depth_colorize_with_mask
            
            source, refined = depth_colorize_with_mask(source_depth[None,:,:],dmindmax=(0.0,5.0)).squeeze() , depth_colorize_with_mask(depthmap[None,:,:], dmindmax=(20.0, 130.0)).squeeze() 
            cv2.imwrite(f"debug/{idx:03d}_source.png", (source[:,:,::-1]*255).astype(np.uint8))
            cv2.imwrite(f"debug/{idx:03d}_refined.png", (refined[:,:,::-1]*255).astype(np.uint8))
            cv2.imwrite(f"debug/{idx:03d}_target.png", target)
            ##########################################################
            
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depthmap, depth_weight=depth_weight,
                              image_path=image_path, image_name=image_name, width=width, height=height, depthloss=depthloss)
        cam_infos.append(cam_info)
        torch.cuda.empty_cache()

    sys.stdout.write('\n')
    return cam_infos

def optimize_depth(source, target, mask, depth_weight, prune_ratio=0.001):
    """
    Arguments
    =========
    source: np.array(h,w)
    target: np.array(h,w)
    mask: np.array(h,w):
        array of [True if valid pointcloud is visible.]
    depth_weight: np.array(h,w):
        weight array at loss.
    Returns
    =======
    refined_source: np.array(h,w)
        literally "refined" source.
    loss: float
    """
    source = torch.from_numpy(source).cuda()
    target = torch.from_numpy(target).cuda()
    mask = torch.from_numpy(mask).cuda()
    depth_weight = torch.from_numpy(depth_weight).cuda()

    pdb.set_trace()

    # Prune some depths considered "outlier"     
    with torch.no_grad():
        target_depth_sorted = target[target>1e-7].sort().values
        min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*prune_ratio)]
        max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*(1.0-prune_ratio))]

        mask2 = target > min_prune_threshold
        mask3 = target < max_prune_threshold
        mask = torch.logical_and( torch.logical_and(mask, mask2), mask3)

    pdb.set_trace()

    source_masked = source[mask]
    target_masked = target[mask]
    depth_weight_masked = depth_weight[mask]
    # tmin, tmax = target_masked.min(), target_masked.max()

    # # Normalize
    # target_masked = target_masked - tmin 
    # target_masked = target_masked / (tmax-tmin)

    pdb.set_trace()

    scale = torch.ones(1).cuda().requires_grad_(True)
    shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

    optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
    loss = torch.ones(1).cuda() * 1e5

    iteration = 1
    loss_prev = 1e6
    loss_ema = 0.0
    
    while abs(loss_ema - loss_prev) > 1e-5:
        source_hat = scale*source_masked + shift
        loss = torch.mean(((target_masked - source_hat)**2)*depth_weight_masked)

        # penalize depths not in [0,1]
        loss_hinge1 = loss_hinge2 = 0.0
        if (source_hat<=0.0).any():
            loss_hinge1 = 2.0*((source_hat[source_hat<=0.0])**2).mean()
        # if (source_hat>=1.0).any():
        #     loss_hinge2 = 0.3*((source_hat[source_hat>=1.0])**2).mean() 
        
        loss = loss + loss_hinge1 + loss_hinge2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        iteration+=1
        if iteration % 1000 == 0:
            print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            loss_prev = loss.item()
        loss_ema = loss.item() * 0.2 + loss_ema * 0.8

    loss = loss.item()
    print(f"loss ={loss:10.5f}")

    with torch.no_grad():
        refined_source = (scale*source + shift) 
    torch.cuda.empty_cache()
    return refined_source.cpu().numpy(), loss


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T               if 'x' in vertices else None
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0 if 'red' in vertices else None
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T              if 'nx' in vertices else None
    errors = vertices['xyzerr']/(np.min(vertices['xyzerr'] + 1e-8))                      if 'xyzerr' in vertices else None
    return BasicPointCloud(points=positions, colors=colors, normals=normals, errors=errors)

def storePly(path, xyz, rgb, xyzerr=None):
    print(f'[Debug] | path = {path}, xyz.shape = {xyz.shape}, rgb.shape = {rgb.shape}')
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('xyzerr', 'f4')]
    
    normals = np.zeros_like(xyz)
    if xyzerr is None:
        xyzerr = np.ones((xyz.shape[0],1))

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, xyzerr), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def refineColmapWithIndex(path, train_index):
    """ result
    'cam_extrinsics' and 'point3D.ply' contains the points observed in (at least 2) train-views 
    """
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    
    xyz, rgb, err = read_points3D_binary(bin_path)
    
    total_ptsidxlist, train_ptsidxlist = [], []
    for tidx, key in enumerate(sorted(cam_extrinsics, key=lambda x:cam_extrinsics[x].name)):
        total_ptsidxlist.append(cam_extrinsics[key].point3D_ids) # cam_extrinsics number starts from 1
        if tidx in train_index: 
            train_ptsidxlist.append(cam_extrinsics[key].point3D_ids)
    
    ### valid 2D points (select the points in train-view)
    ptsidx, cnt = np.unique(np.concatenate(train_ptsidxlist), return_counts=True) # for 2D points (extr.xys, extr.point3D_ids)
    
    valid_ptsidx = ptsidx[cnt>=min(2, len(train_index))][1:] # 2view -> 3view: more restrict condition (COLMAP uses 3-view observed feature points)
    
    for tidx, key in enumerate(sorted(cam_extrinsics, key=lambda x:cam_extrinsics[x].name)):
        cam_valid = np.isin(cam_extrinsics[key].point3D_ids, valid_ptsidx)
        cam_extrinsics[key] = cam_extrinsics[key]._replace(point3D_ids = cam_extrinsics[key].point3D_ids[cam_valid],
                                                                 xys =  cam_extrinsics[key].xys[cam_valid])

    ### valid 3D points (removing the points only detected in 1 camera)
    ptsidx, cnt = np.unique(np.concatenate(total_ptsidxlist), return_counts=True) # for 3D points (xyz, rgb, err from points3D.bin)
    valid_totalptsidx = ptsidx[cnt>=min(2, len(train_index))][1:] # remove invalid(-1) pts
    assert len(valid_totalptsidx)==len(xyz)
    
    valid3didx = np.isin(valid_totalptsidx, valid_ptsidx) # select the points seen from train-view
    xyz = xyz[valid3didx]
    rgb = rgb[valid3didx]
    err = err[valid3didx]
    
    ### save in ply format
    os.makedirs(os.path.join(path, 'plydummy'), exist_ok=True)
    ply_path = os.path.join(path, 'plydummy', f"points3D_{int(datetime.now().timestamp())}.ply") # k-shot train
    storePly(ply_path, xyz, rgb, xyzerr=err)
    
    return ply_path, cam_extrinsics, cam_intrinsics


def transformed_gtcams(gtcams, tarcam5):
    gtcam5 = {}
    gtcam5_R, gtcam5_T = [], []
    tarcam5_R, tarcam5_T = [], []
    for targetkey in tarcam5.keys():
        tarcam5[targetkey].name
        for gtkey in gtcams.keys(): 
            if tarcam5[targetkey].name == gtcams[gtkey].name:
                gtcam5[targetkey] = gtcams[gtkey]
                gtcam5_R.append(qvec2rotmat(gtcam5[targetkey].qvec))
                gtcam5_T.append(gtcam5[targetkey].tvec)
                
                tarcam5_R.append(qvec2rotmat(tarcam5[targetkey].qvec))
                tarcam5_T.append(tarcam5[targetkey].tvec)
    gtcam5_R, gtcam5_T = torch.tensor(np.array(gtcam5_R)), torch.tensor(np.array(gtcam5_T))
    tarcam5_R, tarcam5_T = torch.tensor(tarcam5_R), torch.tensor(tarcam5_T)

    def get_relative_cam1cam2(cam_1, cam_2):
        trans_rel = cam_1.inverse().compose(cam_2)
        matrix_rel = trans_rel.get_matrix()
        cams_relative = SfMPerspectiveCameras(
                        R = matrix_rel[:, :3, :3],
                        T = matrix_rel[:, 3, :3])
        return cams_relative


    tar_gt_rel0 = get_relative_cam1cam2(SfMPerspectiveCameras(R=tarcam5_R, T=tarcam5_T).get_world_to_view_transform(),
                                        SfMPerspectiveCameras(R=gtcam5_R, T=gtcam5_T).get_world_to_view_transform())

    targtrel_R = so3_exp_map(so3_log_map(tar_gt_rel0.R).mean(dim=0, keepdim=True)).clone().detach()
    targtrel_T = tar_gt_rel0.T.mean(dim=0, keepdim=True).clone().detach()
    
    R = torch.matmul(tarcam5_R, targtrel_R.double())
    T = tarcam5_T + targtrel_T.double()
    
    ### update
    for idx, targetkey in enumerate(tarcam5.keys()):
        tarcam5[targetkey].name
        for gtkey in gtcams.keys(): 
            if tarcam5[targetkey].name == gtcams[gtkey].name:
                gtcams[gtkey]._replace(qvec=rotmat2qvec(R[idx].numpy()), tvec=T[idx].numpy())
    
    return gtcams, targtrel_R, targtrel_T
    

def BundleAdjustment_ColmapWithIndex(path, train_index):
    """ result
    path: tmp_fewshot
    'cam_extrinsics' and 'point3D.ply' contains the points observed in (at least 2) train-views 
    """
    try:
        tmp_fd = "tmp_fewshot"+str(int(datetime.now().timestamp()))
        
        os.makedirs(os.path.join(path, tmp_fd, "images"), exist_ok=True) #
        for idx in train_index:
            imgs1 = os.path.join(path, 'images', f'{idx:05d}.jpg')
            imgs2 = os.path.join(path, tmp_fd, 'images', f'{idx:05d}.jpg')
            os.system(f"cp {imgs1} {imgs2}")
        
        os.system(f"colmap automatic_reconstructor --workspace_path {os.path.join(path, tmp_fd)} --image_path {os.path.join(path, tmp_fd, 'images')} --camera_model PINHOLE --single_camera 1 --dense 0 --num_threads 8") #
        
        cam_extrinsics = read_extrinsics_binary(os.path.join(path, "sparse/0", "images.bin"))
        tarcam5 = read_extrinsics_binary(os.path.join(path, tmp_fd, "sparse/0", "images.bin")) 

        cam_extrinsics, targtrel_R, targtrel_T = transformed_gtcams(cam_extrinsics, tarcam5)
        
        xyz, rgb, err = read_points3D_binary(os.path.join(path, tmp_fd, "sparse/0/points3D.bin"))
        xyz = (torch.tensor(xyz).float()@targtrel_R[0] + targtrel_T[0]).numpy()
        
        ### save in ply format
        os.makedirs(os.path.join(path, 'plydummy'), exist_ok=True)
        ply_path = os.path.join(path, 'plydummy', f"points3D_{int(datetime.now().timestamp())}.ply") # k-shot train
        storePly(ply_path, xyz, rgb, xyzerr=err)
    
        os.system(f"rm -r {os.path.join(path, tmp_fd)}") #
    
    except:
        os.system(f"rm -r {os.path.join(path, tmp_fd)}") #
        raise("CANNOT RUN COLMAP!!")
    
    cam_intrinsics = read_intrinsics_binary(os.path.join(path, "sparse/0", "cameras.bin"))
    
    return ply_path, cam_extrinsics, cam_intrinsics


def unprojFullDepth(caminfo, ply_path, single=False):
    H, W = caminfo[0].depth.shape
    FX, FY = fov2focal(caminfo[0].FovX, W), fov2focal(caminfo[0].FovY, H)
    K = np.array([[FX,0,W/2],[0,FY,H/2],[0,0,1]])
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
    positions, colors = [], []
    for i in range(len(caminfo)):
        z, R, T = caminfo[i].depth, caminfo[i].R, caminfo[i].T.reshape(3,1)
        pts_cam = np.matmul(np.linalg.inv(K), np.stack((x*z, y*z, 1*z), axis=0).reshape(3,-1))
        pts_world = (R.dot(pts_cam) - R.dot(T)).astype(np.float32)
        pts_color = np.array(caminfo[i].image).reshape(-1,3)
        
        
        ##### DepthInitValid
        valid_i_total = (np.ones_like(caminfo[i].depth).reshape(-1)>0)
        for j in range(len(caminfo)):
            if j!=i:
                pts_cam_j = np.matmul(K, np.matmul(caminfo[j].R.transpose(), pts_world) + caminfo[j].T.reshape(3,1)) # (3, -1)
                valid_idx_j = np.logical_and.reduce((pts_cam_j[2]>0, pts_cam_j[0]/pts_cam_j[2]>=0, pts_cam_j[0]/pts_cam_j[2]<=W-1, pts_cam_j[1]/pts_cam_j[2]>=0, pts_cam_j[1]/pts_cam_j[2]<=H-1))
                depth_j = pts_cam_j[-1:, valid_idx_j]
                pts_cam_j = np.round(pts_cam_j[:2, valid_idx_j]/pts_cam_j[-1:, valid_idx_j]).astype(np.int32)
                
                color_diff = np.abs(np.array(caminfo[j].image)[pts_cam_j[1], pts_cam_j[0]] - pts_color[valid_idx_j]).mean(axis=1) <= 1
                depth_diff = np.abs(caminfo[j].depth[pts_cam_j[1], pts_cam_j[0]] - depth_j)[0] < 0.01
                
                valid_idx_j[valid_idx_j==True] = np.logical_and(color_diff, depth_diff)
                
                valid_i_total = np.logical_and(valid_i_total, valid_idx_j)
        ###################
        pts_color = np.array(caminfo[i].image.resize(z.shape[::-1])).reshape(-1,3)
        positions.append(pts_world.transpose(1,0)[valid_i_total])
        colors.append(pts_color[valid_i_total])
        
        if single:
            break
        
    positions, colors = np.concatenate(positions), np.concatenate(colors)
    randidx = np.random.choice(np.arange(len(positions)), size=min(len(positions)//len(caminfo)*2, 100_000), replace=False)
    positions, colors = positions[randidx], colors[randidx]
    
    storePly(ply_path, positions, colors)
    return BasicPointCloud(points=positions, colors=colors, normals=None, errors=None)


def pick_idx_from_360(path, train_idx, kshot, center, num_trials=100_000):
    """
    [Taekkii]
    randomly pick ONE index from train_idx.
    The rest are decided by RANSAC-like brute search method to match criterion:
    - let vectors v: center to each camera positions.
    - maximize: prod(angle between two vectors)

    ARGUMENTS
    ---------
    path: colmap standard directory path.
    train_idx: list of train indice.
    kshot: # of shots (int)
    center: ndarray(3): center saved in the path.
    num_trials: number of RANSAC search trials.

    RETURNS
    -------
    indice: (list)selected train-indice. Not guaranteed to be optimal.
            NOTE: Same seed always results in same indice.
    """
    
    # guard.
    if kshot>=len(train_idx):
        return train_idx
    
    # Get camera positions. Kinda redundant code, but we read extrinsics again.
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    
    cam_locs = []
        
    for idx, key in enumerate(sorted(cam_extrinsics, key=lambda x:cam_extrinsics[x].name)):
        
        if idx not in train_idx:
            continue
        
        cam_extrinsic = cam_extrinsics[key]
        R = np.transpose(qvec2rotmat(cam_extrinsic.qvec))
        T = np.array(cam_extrinsic.tvec)
        cam_locs.append(-T@R)

    cam_locs = np.stack(cam_locs)

    # fix pivot index.
    pivot = np.random.randint(len(train_idx))
    
    choice_indice_pull = np.array([e for e in range(len(train_idx)) if e != pivot])
    candidate_indice, candidate_criterion = None , 0.0

    pivot = np.array([pivot])

    # RANSAC-like random search.
    for _ in tqdm(range(num_trials), desc='Choosing best indice'):
        indice = np.random.choice(choice_indice_pull, kshot-1, replace=False)
        indice = np.concatenate([indice,pivot]) # Always include pivot.

        selected_camlocs = cam_locs[indice] # (kshot,3)
        vectors = selected_camlocs - center # (kshot,3)
        
        # Take unit vector (makes my life easier.)
        vectors = vectors / np.linalg.norm(vectors,axis=-1,keepdims=True)
        
        radians = np.arccos( (vectors * vectors[:,None,:]).sum(axis=-1).clip(-0.99999 , 0.99999) ) # (kshot,3) * (kshot,1,3) --> (kshot,kshot,3) --(sum)--> (kshot,kshot) 

        criterion = radians.prod() # Strictly speaking, sqrt of this criterion.

        if candidate_criterion < criterion:
            candidate_criterion = criterion
            candidate_indice = indice

    final_indice = (np.array(train_idx)[candidate_indice]).tolist()
    return final_indice


def readColmapSceneInfo(path, images, eval, kshot=1000, seed=0, resolution=4, white_background=False):
    ## load split_idx.json 
    with open(os.path.join(path, "split_index.json"), "r") as jf:
        jsonf = json.load(jf)
        train_idx, test_idx = jsonf["train"], jsonf["test"]
    
    reading_dir = "images" if images == None else images

    scene_center_path = os.path.join(path, "center.npy")
    
    np.random.seed(seed)
    if os.path.exists(scene_center_path) and eval:
        train_idx = pick_idx_from_360(path, train_idx, kshot, center=np.load(scene_center_path))
    else:
        train_idx = sorted(np.random.choice(train_idx, size=min(kshot, len(train_idx)), replace=False)) if eval else np.arange(len(train_idx)).tolist()
    
    ### refineColmapWithIndex() remove the cameras and features except the train set
    ply_path, cam_extrinsics, cam_intrinsics = refineColmapWithIndex(path, train_idx)
    
    ### making pcd with the features captured from train_cam
    pcd = fetchPly(ply_path)
    
    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                  images_folder=os.path.join(path, reading_dir), pcd=pcd, resolution=resolution, train_idx=train_idx, white_background=white_background).copy()

    if eval:
        train_cam_infos = [cam_infos[i] for i in train_idx]
        test_cam_infos = [cam_infos[i] for i in test_idx]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []     

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# def depth_minmax(depth_name):
#     depths = np.stack(depths)
#     batch, vx, vy = np.where(depths!=0)
#
#     valid_depth = depths[batch, vx, vy]
#     return valid_depth.min(), valid_depth.max()
#

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", use_depth=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        
        
        depth_namelist = sorted(glob(os.path.join(path, '/'.join(frames[0]["file_path"].split('/')[:-1])+ "/*depth*")))
        if len(depth_namelist)>0:
            depths = []
            for i in range(len(depth_namelist)):
                depths.append(np.load(depth_namelist[i])) # normalized [0,1]
            depths = np.stack(depths)
            batch, vx, vy = np.where(depths!=0)
                
            valid_depth = depths[batch, vx, vy]
            dmin, dmax = valid_depth.min(), valid_depth.max()

        
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if use_depth and os.path.exists(os.path.join(path, frame["file_path"].replace("/train/", "/depths_train/")+'.npy')):
                depth = np.load(os.path.join(path, frame["file_path"].replace("/train/", "/depths_train/")+'.npy'))
                if os.path.exists(os.path.join(path, frame["file_path"].replace("/train/", "/masks_train/")+'.png')):
                    mask = imageio.v3.imread(os.path.join(path, frame["file_path"].replace("/train/", "/masks_train/")+'.png'))[:,:,0]/255.
                else:
                    mask = np.ones_like(depth)
                final_depth = depth*mask
            else:
                final_depth = None

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=final_depth,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], depthloss=None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", use_depth=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, use_depth=use_depth)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, use_depth=use_depth)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



# class Dair_v2x_Info(NamedTuple):
#     # in LiDAR coordinate
#     side_type: str                      # inf / veh
#     pcd: BasicPointCloud
#     rgb_gt: dict                        # fg/bg/panoptic: np.array [H W 3]
#     colored_depth: dict                 # colored, fg/bg/panoptic: np.array [H W] ? [H W 1] ?
#     colored_init_depth: dict            # colored, fg/bg/panoptic: ...
#     pred_depth: dict                    # no colore, fg/bg/panoptic: ...
#     to_inf: np.array

class Dair_v2x_Info(NamedTuple):
    # in LiDAR coordinate
    inf_pcd: BasicPointCloud
    veh_pcd: BasicPointCloud
    inf_rgb: np.array
    veh_rgb: np.array
    inf_depth: dict
    veh_depth: dict
    inf_cam_K: dict
    veh_cam_K: dict
    inf2veh_matrix: np.array
    lidar2cam_inf: np.array
    lidar2cam_veh: np.array
    world2cam_inf: np.array
    world2cam_veh: np.array

    normalization_inf: dict
    normalization_veh: dict

    """
    
    Original Scene_Info
        point_cloud: BasicPointCloud
        train_cameras: list
        test_cameras: list
        nerf_normalization: dict
        ply_path: str
    
    """

def AABB_func(cam_list: list) -> dict:
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers) # [3 1]*n -> [3 n]
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True) # -> mean -> [3 n]
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True) # Euclidean distance
        diagonal = np.max(dist) # 包围相机的AABB框的对角线长度
        return center.flatten(), diagonal

    cam_center = [cam[:3, 3:4] for cam in cam_list]
    center, diagonal = get_center_and_diag(cam_center) # input: [..., ...]
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}





def readDairV2XSyntheticInfo(
        pair: CooperativeData = None,
        inf_side_info: dict = None,
        veh_side_info: dict = None
):
    # pre-read DAIR-V2X dataset
    # dair_item: CooperativeData
    # inf_idx, veh_idx = pair.inf_id, pair.veh_id

    inf_cam_K = pair.load_intrinsic(pair.inf_cam_intrinsic_path) # h, w, K[3,3]
    veh_cam_K = pair.load_intrinsic(pair.veh_cam_intrinsic_path, matrix_read_only=True) # h, w, K[3,3]

    lidar2cam_inf, lidar2cam_veh = pair.load_extrinsic(pair.inf_lidar2cam_path), pair.load_extrinsic(pair.veh_lidar2cam_path)
    inf2veh = np.linalg.inv(lidar2cam_inf) @ lidar2cam_veh

    lidar2world_inf = pair.load_extrinsic(pair.inf_lidar2world_path)
    lidar2world_veh = pair.load_extrinsic(pair.veh_lidar2novatel_path) @ pair.load_extrinsic(pair.veh_novatel2world_path)

    world2cam_inf = np.linalg.inv(lidar2world_inf) @ lidar2cam_inf # [4 4]
    world2cam_veh = np.linalg.inv(lidar2world_veh) @ lidar2cam_veh # [4 4]
    inf_pcd, veh_pcd = np.asarray(inf_side_info['pcd'].points), np.asarray(veh_side_info['pcd'].points)
    # -> [X 3]
    # inf_pcd, veh_pcd = lidar2world_inf @ inf_pcd, lidar2world_veh @ veh_pcd # transfer to world coordinate
    print(f'[DEBUG] <in readDairV2XSyntheticInfo>: type(inf_pcd) = {type(inf_pcd)}, type(veh_pcd) = {type(veh_pcd)}')
    # .points ✅ .colors ❎ .normals ❎
    print(f'[DEBUG] inf_pcd.shape = {inf_pcd.shape}, veh_pcd.shape = {veh_pcd.shape}')

    # We create random points inside the bounds of the synthetic Blender scenes
    inf_rgb = SH2RGB(np.random.random((inf_pcd.shape[0], 3)) / 255.) # -> ([X 3])
    veh_rgb = SH2RGB(np.random.random((veh_pcd.shape[0], 3)) / 255.) # ([X 3])

    # inf_pcd_ = BasicPointCloud(points=inf_pcd, colors=inf_rgb, normals=np.zeros((inf_pcd.shape[1],3)))
    # veh_pcd_ = BasicPointCloud(points=veh_pcd, colors=veh_rgb, normals=np.zeros((veh_pcd.shape[1], 3)))
    storePly(pair.inf_ply_store_path, inf_pcd, inf_rgb * 255)
    storePly(pair.veh_ply_store_path, veh_pcd, veh_rgb * 255)

    normalized_inf = AABB_func([world2cam_inf])
    normalized_veh = AABB_func([world2cam_veh])

    return Dair_v2x_Info(
                inf_pcd = inf_side_info['pcd'], veh_pcd = veh_side_info['pcd'],
                inf_rgb = np.array(inf_side_info['rgb']), veh_rgb = np.array(veh_side_info['rgb']),
                inf_depth = inf_side_info['depth'],       veh_depth = veh_side_info['depth'],
                inf_cam_K = {'height': inf_cam_K[0], 'width': inf_cam_K[1], 'cam_K': inf_cam_K[2]},
                veh_cam_K = {'height': inf_cam_K[0], 'width': inf_cam_K[1], 'cam_K': veh_cam_K},
                inf2veh_matrix = inf2veh,  # [4 4],
                lidar2cam_inf = lidar2cam_inf, lidar2cam_veh = lidar2cam_veh,
                world2cam_inf = world2cam_inf, world2cam_veh = world2cam_veh,
                normalization_inf = normalized_inf, normalization_veh = normalized_veh
            )



sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "V2X": readDairV2XSyntheticInfo
}



def CreateCamera(
        dair_item: CooperativeData, dair_info: Dair_v2x_Info,
        side_info: dict,
        type: str = 'inf', white_background: bool = False
    ):
    pcd_file = side_info['pcd']
    uid = str(getattr(dair_item, f'{type}_id'))
    bg = np.array([1, 1, 1, 0]) if white_background else np.array([0, 0, 0, 0])

    cam_intrinsic = getattr(dair_info, f'{type}_cam_K') # dict
    cam_K = cam_intrinsic['cam_K'] # 'K_matrix'
    height, width = cam_intrinsic['height'], cam_intrinsic['width']
    cam_extrinsic = getattr(dair_info, f'lidar2cam_{type}')  # np.array

    R, T = cam_extrinsic[0:3,0:3], cam_extrinsic[0:3, 3]
    FovX, FovY = cam_K[0,0], cam_K[1,1]
    depth_map, depth_weight = np.zeros((height, width)), np.zeros((height, width))
    cam_coord = np.matmul(cam_K, np.matmul(R.transpose(), np.asarray(pcd_file.points).transpose()) + T.reshape(3, 1))
    ### for coordinate definition, see getWorld2View2() function
    valid_idx = np.where(np.logical_and.reduce((cam_coord[2] > 0, cam_coord[0] / cam_coord[2] >= 0,
                                                cam_coord[0] / cam_coord[2] <= width - 1,
                                                cam_coord[1] / cam_coord[2] >= 0,
                                                cam_coord[1] / cam_coord[2] <= height - 1)))[0]
    pts_depths = cam_coord[-1:, valid_idx]
    cam_coord = cam_coord[:2, valid_idx] / cam_coord[-1:, valid_idx]
    depth_map[np.round(cam_coord[1]).astype(np.int32).clip(0, height - 1), np.round(cam_coord[0]).astype(np.int32).clip(0, width - 1)] = pts_depths
    depth_weight[np.round(cam_coord[1]).astype(np.int32).clip(0, height - 1), np.round(cam_coord[0]).astype(np.int32).clip(0, width - 1)] = 1
    depth_weight = depth_weight / depth_weight.max()

    rgb_img, source_depth = side_info['rgb'], side_info['depth']['panoptic'][0] # 0, 1, 2 ?

    target = depth_map.copy()
    target = ((target != 0) * 255).astype(np.uint8) # mask

    pdb.set_trace()
    source_depth = np.mean(source_depth, axis=-1)
    print(f'[DEBUG] source_depth = {source_depth.shape}, depth_map.shape = {depth_map.shape}, depth_weight.shape = {depth_weight.shape}')

    depth_map, depthloss = optimize_depth(source=source_depth, target=depth_map, mask=(depth_map > 0.), depth_weight=depth_weight)

    # import cv2
    # from render import depth_colorize_with_mask
    #
    # source, refined = depth_colorize_with_mask(source_depth[None, :, :],
    #                                            dmindmax=(0.0, 5.0)).squeeze(), depth_colorize_with_mask(
    #     depthmap[None, :, :], dmindmax=(20.0, 130.0)).squeeze()

    pdb.set_trace()

    # BUG - BEGIN

    print(f'depth_map.shape = {depth_map.shape}')
    if not os.path.exists('./debug'): os.mkdir('./debug')
    pred_init = depth_map.squeeze().detach().cpu().numpy().astype(np.uint8)
    M, m = np.max(pred_init), np.min(pred_init)
    pred_init = (pred_init - m) / (M - m) * 255.
    colored_init = cv2.applyColorMap(pred_init.astype(np.uint8), cv2.COLORMAP_RAINBOW)

    # BUG - END

    pdb.set_trace()

    source_path = '/debug/sources'
    if not os.path.exists(source_path): os.mkdir(source_path)

    print(f'[Debug] (colored_init[:, :, ::-1] * 255).astype(np.uint8).shape = {(colored_init[:, :, ::-1] * 255).astype(np.uint8).shape}')
    print(f'[Debug] target.shape = {target.shape}')

    cv2.imwrite(os.path.join(source_path, f"{uid:03d}_source.png"), (colored_init[:, :, ::-1] * 255).astype(np.uint8))
    # cv2.imwrite(f"./debug/{uid:03d}_refined.png", (refined[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(source_path, f"{uid:03d}_target.png"), target)

    pdb.set_trace()

    return CameraInfo(uid=str(0 if type=='inf' else 1) + uid, R=R, T=T, FovY=FovY, FovX=FovX, image=rgb_img, depth=depth_map,
               depth_weight=depth_weight, image_path=getattr(dair_item, f'{type}_img_path'),
               image_name=uid, width=width, height=height, depthloss=depthloss)


class Scene:

    # gaussians : GaussianModel
    # multi Gaussian -> i.e., using multi class<Scene>

    def __init__(
            self, args: ModelParams, dair_item: CooperativeData,
            dair_info: Dair_v2x_Info, gaussians,
            type: str='inf', seed=0, side_info: dict=None,
            shuffle: bool=True, eval: bool = False,
            resolution_scales=[1.0]
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = dair_item.model_path # dair path
        self.teyp = type
        anti_type = 'inf' if type == 'veh' else 'veh'
        self.gaussians = gaussians

        """
            Dair_v2x_Info(
                inf_pcd = inf_pcd_, veh_pcd = veh_pcd_,
                inf_rgb = np.array(inf_side_info['rgb']), veh_rgb = np.array(veh_side_info['rgb']),
                inf_depth = inf_side_info['depth'],       veh_depth = veh_side_info['depth'],
                inf_cam_K = {'h': inf_cam_K[0], 'w': inf_cam_K[1], 'cam_K': inf_cam_K[2]},
                veh_cam_K = {'h': veh_cam_K[0], 'w': veh_cam_K[1], 'cam_K': veh_cam_K[2]},
                inf2veh_matrix = inf2veh,  # [4 4],
                lidar2cam_inf = lidar2cam_inf, lidar2cam_veh = lidar2cam_veh,
                world2cam_inf = world2cam_inf, world2cam_veh = world2cam_veh,
                normalization_inf = normalized_inf, normalization_veh = normalizedd_veh
            )
        """

        # TODO: << Regularize Camera Infos >>
        self.cameras_extent = getattr(dair_info, f'normalization_{type}')["radius"] # 处理好的相机内参
        # Origin: nerf_normalization = {"translate": translate, "radius": radius}

        # TODO: 对dair_info中的veh&inf点云拼接后做语义切割（或者切割后分别拼接），初始化fg/bg GS
        # panoptic_pcd, inf_pcd, veh_pcd = Bind_v2x_pcd(dair_info)
        self.gaussians.create_from_pcd(getattr(dair_info, f'{type}_pcd'), self.cameras_extent)

        # cameraList_from_camInfos
        cam_infos = [CreateCamera(dair_item, dair_info, side_info, type_) for type_ in [type, anti_type]]
        train_cam_infos = [cam_infos[0]] if eval else cam_infos # 当前type必然参与训练
        test_cam_infos = [cam_infos[1]] if eval else []


        np.random.seed(seed)
        scene_info = SceneInfo(point_cloud=getattr(dair_info, f'{type}_pcd'),
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=self.cameras_extent,
                           ply_path=getattr(dair_item, f'{type}_ply_store_path')
                        )

        self.train_cameras, self.test_cameras = {}, {}
        # self.train_cameras[]

        for scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[scale] = cameraList_from_camInfos(scene_info.train_cameras, scale, args)
            print("Loading Test Cameras")
            self.test_cameras[scale] = cameraList_from_camInfos(scene_info.test_cameras, scale, args)
            # cameraList_from_camInfos(?, scale, args)
            # self.train_cameras[scale] = {
            #     'intrinsics': getattr(dair_info, f'normalization_{type}'),
            #     'world2cam': getattr(dair_info, f'world2cam_{type}'),
            #     'lidar2cam': getattr(dair_info, f'lidar2cam_{type}'),
            #     'original_image': getattr(dair_info, f'{type}_rgb'),
            #     'original_depth': getattr(dair_info, f'{type}_depth'),
            #     f'{type}2{anti_type}': dair_info.inf2veh_matrix if type=='inf' else np.lialg.inv(dair_info.inf2veh_matrix)
            # }
            # self.test_cameras[scale] = {
            #     'intrinsics': getattr(dair_info, f'normalization_{anti_type}'),
            #     'world2cam': getattr(dair_info, f'world2cam_{anti_type}'),
            #     'lidar2cam': getattr(dair_info, f'lidar2cam_{anti_type}'),
            #     'original_image': getattr(dair_info, f'{anti_type}_rgb'),
            #     'original_depth': getattr(dair_info, f'{anti_type}_depth'),
            #     f'{anti_type}2{type}': dair_info.inf2veh_matrix if type == 'inf' else np.lialg.inv(dair_info.inf2veh_matrix)
            # }


# , json_cams, self.gaussian.load_ply,

    def save(self, iteration):
        assert os.path.exists(self.model_path)
        a = os.path.join(self.model_path, 'point_cloud')
        if not os.path.exists(a): os.mkdir(a)
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, f"pcd-{self.type}.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]







def returnMultiPCD(inf_scene: Scene, veh_scene: Scene) -> Scene:




    # setattr(.., temp_camera(...))
    ...

def returnConbinedPCD(inf_scene: Scene, veh_scene: Scene) -> Scene:
    # 在世界坐标系下合并点云, 使用inf_view_veh和veh_view_inf
    # 其实不需要合并点云，只用对侧的view即可
    ...


def returnConbinedDepth(inf_scene: Scene, veh_scene: Scene) -> Scene:
    ...

def returnDirectlyV2X(scene: Dair_v2x_Info) -> Scene:
    ...

sceneConbinationCallbacks = {
    'multi-pcd': returnMultiPCD, # 多点云, 作为视角list
    'conbine-pcd': returnConbinedPCD, # 先合并点云, 两个深度
    'conbine-depth': returnConbinedDepth, # 点云不合并, 使用深度时对深度用系数omit(或者再开新的)叠加
    'deal-v2x': returnDirectlyV2X # 直接使用v2x-pair
}


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

import os
import random
import json

import numpy as np

from ..utils.system_utils import searchForMaxIteration
from ..scene.dataset_readers import sceneLoadTypeCallbacks
from ..scene.gaussian_model import GaussianModel
from .. import ModelParams
from ..utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from lidar2dep.dair import CooperativeData
from ..scene.dataset_readers import Dair_v2x_Info, BasicPointCloud

def Bind_v2x_pcd(dair: Dair_v2x_Info, inference_mode=False) -> tuple[BasicPointCloud]:
    # 已经是世界坐标系，应该不用做poing cloud registration，直接拼接
    # align infrastructure side PCD with vehicle side PCD
    # TODO: Point Cloud Registration
    """
    Inference Mode:
    「
        Infrastructure Side:    lidar-inf-pcd -> world-inf-pcd
        Vehicle Side:           lidar-veh-pcd -> novatel-veh-pcd -> world-veh-pcd

                =>  Point Clout Registration => △p = [R|t]

        lidar-veh-pcd -> cam-veh-pcd --△p--> cam-inf-pcd
                                                                」

    On DAIR-V2X Dataset:
        △p: dair.inf2veh

    """

    # 已经在world坐标系下， 直接合并出新点云
    inf_pcd, veh_pcd = dair.inf_pcd, dair.veh_pcd
    bind_axis = 0 if inf_pcd.points.shape[0] > inf_pcd.points.shape[1] else 1


    return BasicPointCloud(
                points = np.concatenate([inf_pcd.points, veh_pcd.points], aixs=bind_axis),
                colors = np.concatenate([inf_pcd.colors, veh_pcd.colors], axis=bind_axis),
                normals = np.concatenate([inf_pcd.colors, veh_pcd.colors], axis=bind_axis)
            ), inf_pcd, veh_pcd





class Scene:

    gaussians : GaussianModel

    def __init__(self, dair_item: CooperativeData, gaussians : list[GaussianModel], inf_side_info: dict=None, veh_side_info: dict=None, shuffle: bool=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        assert len(gaussians) >= 3, gaussians
        self.model_path = dair_item.model_path # dair path
        self.gaussians_inf = gaussians[0]
        self.gaussians_veh = gaussians[1]
        self.gaussians_panoptic = gaussians[2]

        dair_info = sceneLoadTypeCallbacks['V2X'](dair_item) # lidar coordinate -> world coordinate

        """
            Dair_v2x_Info(
                inf_pcd=inf_pcd_, veh_pcd=veh_pcd_,
                inf_rgb=np.array(inf_side_info['rgb']), veh_rgb=np.array(veh_side_info['rgb']),
                inf_depth=inf_side_info['depth'], veh_depth=veh_side_info['depth'],
                inf_cam_K={'h': inf_cam_K[0], 'w': inf_cam_K[1], 'cam_K': inf_cam_K[2]},
                veh_cam_K={'h': veh_cam_K[0], 'w': veh_cam_K[1], 'cam_K': veh_cam_K[2]},
                inf2veh_matrix=inf2veh  # [4 4]
            )
        """

        # TODO: << Regularize Camera Infos >>
        self.cameras_extent = scene_info.nerf_normalization["radius"] # 处理好的相机内参
        # Origin: nerf_normalization = {"translate": translate, "radius": radius}

        # TODO: 对dair_info中的veh&inf点云拼接后做语义切割（或者切割后分别拼接），初始化fg/bg GS
        panoptic_pcd, inf_pcd, veh_pcd = Bind_v2x_pcd(dair_info)
        self.gaussians_veh.create_from_pcd(veh_pcd, self.cameras_extent)
        self.gaussians_inf.create_from_pcd(inf_pcd, self.cameras_extent)
        self.gaussians_panoptic.create_from_pcd(panoptic_pcd, self.cameras_extent)



# , json_cams, self.gaussian.load_ply,

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians_panoptic.save_ply(os.path.join(point_cloud_path, "pcd-panoptic.ply"))
        self.gaussians_inf.save_ply(os.path.join(point_cloud_path, "pcd-inf.ply"))
        self.gaussians_veh.save_ply(os.path.join(point_cloud_path, "pcd-veh.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


class Depth_Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path # dataset root path
        self.loaded_iter = None
        self.gaussians = gaussians

        # if load_iteration:
        #     if load_iteration == -1:
        #         self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        #     else:
        #         self.loaded_iter = load_iteration
        #     print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}


        # if os.path.exists(os.path.join(args.source_path, "sparse")):
        #     scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, kshot=args.kshot,
        #                                                   seed=args.seed, resolution=args.resolution,
        #                                                   white_background=args.white_background)
        # elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # else:
        #     assert False, "Could not recognize scene type!"
        scene_info = sceneLoadTypeCallbacks["V2X"]()

        # if not self.loaded_iter:
        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
            dest_file.write(src_file.read())
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    # , json_cams, self.gaussian.load_ply,

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
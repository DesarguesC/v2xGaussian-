from process import process_first
import torch
import numpy as np




"""


sh_dgree = lp.extract(args).sh_degree

gs_model = GaussianModel(sh_degree)
gs_model.create_from_pcd(pcd : BasicPointCloud, spatial_lr_scale : float)
    -> pcd: 「scene_info.point_cloud」就用open3d加载出来的内容？其中也有.points, .colors方法
    -> spatioal_lr_scale: self.cameras_extent 
            -> scene_info.nerf_normalization["radius"]
                -> sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, kshot=args.kshot,
                                                          seed=args.seed, resolution=args.resolution,
                                                          white_background=args.white_background)
                            -> 「 def readColmapSceneInfo(...) -> class SceneInfo 」
    
                                        class SceneInfo(NamedTuple):
                                        point_cloud: BasicPointCloud
                                        train_cameras: list
                                        test_cameras: list
                                        nerf_normalization: dict
                                        ply_path: str

"""




def train_DRGS(opt, ):










def main():
    processed_dict = process_first()





if __name__ == "__main__":
    main()
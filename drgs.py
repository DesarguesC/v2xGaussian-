from process import process_first
import os, sys, uuid, cv2, torch, torchvision
from tqdm import tqdm
from random import randint
import cv2
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from argparse import ArgumentParser, Namespace
import drgs_utils

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


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def train_DRGS(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, usedepth=False, usedepthReg=False):
    # {lp.extract(args), op.extract(args), pp.extract(args)} -> {dataset, opt, pipe} -> 都是传参数成员的
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)  # TODO: transport depth&v2x-scene here
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_depthloss_for_log, prev_depthloss, deploss = 0.0, 1e2, torch.zeros(1)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            if not (usedepth and iteration >= 2000):
                gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        ### depth supervised loss
        depth = render_pkg["depth"]
        if usedepth and viewpoint_cam.original_depth is not None:
            depth_mask = (viewpoint_cam.original_depth > 0)  # render_pkg["acc"][0]
            gt_maskeddepth = (viewpoint_cam.original_depth * depth_mask).cuda()
            if args.white_background:  # for 360 datasets ...
                gt_maskeddepth = normalize_depth(gt_maskeddepth)
                depth = normalize_depth(depth)

            deploss = l1_loss(gt_maskeddepth, depth * depth_mask) * 0.5
            loss = loss + deploss

        ## depth regularization loss (canny)
        if usedepthReg and iteration >= 0:
            depth_mask = (depth > 0).detach()
            nearDepthMean_map = nearMean_map(depth, viewpoint_cam.canny_mask * depth_mask, kernelsize=3)
            loss = loss + l2_loss(nearDepthMean_map, depth * depth_mask) * 1.0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_depthloss_for_log = 0.2 * deploss.item() + 0.8 * ema_depthloss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{ema_loss_for_log:.{7}f}", "Deploss": f"{ema_depthloss_for_log:.4f}",
                     "#pts": gaussians._xyz.shape[0]})
                progress_bar.update(10)

            if iteration % 100 == 0:
                if iteration > opt.min_iters and ema_depthloss_for_log > prev_depthloss:
                    training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                    [iteration], scene, render,
                                    (pipe, background), txt_path=os.path.join(args.model_path, "metric.txt"))
                    scene.save(iteration)
                    print(f"!!! Stop Point: {iteration} !!!")
                    break
                else:
                    prev_depthloss = ema_depthloss_for_log

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background), txt_path=os.path.join(args.model_path, "metric.txt"))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and ((not usedepth) or gaussians._xyz.shape[0] <= 1500000):
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                size_threshold)

                if not usedepth:
                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    pass








def main():
    processed_dict = process_first()





if __name__ == "__main__":
    main()
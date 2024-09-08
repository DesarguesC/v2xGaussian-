from process import process_first
import os, sys, uuid, cv2, torch, torchvision
from tqdm import tqdm
from random import randint
import cv2
from drgs_utils import *
from drgs_utils.scene import sceneLoadTypeCallbacks
import numpy as np

from lidar2dep.dair import DAIR_V2X_C, CooperativeData


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from argparse import ArgumentParser, Namespace


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



def Reporter(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, txt_path=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    depth = render_pkg['depth']

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    # [Taekkii-HARDCODING] Save images.
                    if txt_path is not None and config['name'] == 'test':
                        elements = txt_path.split("/")
                        elements = elements[:-1]
                        render_path = os.path.join(*elements, f"render_{iteration:05d}")
                        os.makedirs(render_path, exist_ok=True)
                        render_image_path = os.path.join(render_path, f"{idx:03d}_render.png")
                        gt_image_path = os.path.join(render_path, f"{idx:03d}_gt.png")
                        torchvision.utils.save_image(image, render_image_path)
                        torchvision.utils.save_image(gt_image, gt_image_path)

                        # depth.
                        depth_path = os.path.join(render_path, f"{idx:03d}_depth.png")
                        depth = ((depth_colorize_with_mask(depth.cpu().numpy()[None])).squeeze() * 255.0).astype(
                            np.uint8)
                        cv2.imwrite(depth_path, depth[:, :, ::-1])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(
                    f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.6f} PSNR {psnr_test:.2f}")  # SSIM {ssim_test:.4f} LPIPS {lpips_test:.4f}")
                if config['name'] == 'test':
                    with open(txt_path, "a") as fp:
                        print(f"{iteration}_{psnr_test:.6f}_{ssim_test:.6f}_{lpips_test:.6f}", file=fp)

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def create_params(parser):
    no_parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(no_parser)
    op = OptimizationParams(no_parser)
    pp = PipelineParams(no_parser)
    return lp.extract(parser), op.extract(parser), pp.extract(parser)


def train_DRGS(
        args, dair_item: CooperativeData,
        inf_side_info: dict, veh_side_info:dict
):
    """

    Args:
        args:
        dair_item:
        inf_side_info:
        veh_side_info:

        xxx_side_info ->
        pred_depth[i]:
            {
                'rgb': image, # pil
                'mask': mask, # fg-mask
                'depth': {
                'fg': (colored_pred_fg, colored_init_fg, pred_fg),
                'bg': (colored_pred_bg, colored_init_bg, pred_bg),
                'panoptic': (colored_pred_all, colored_init, pred)
                }, 'pcd': pcd_file # use open3d.io.read_point_cloud(...)
            }

        -> pred_fg, pred_bg, pred -> uncolored

    Returns:

    """

    gt_depth_inf_dict, gt_depth_veh_dict = inf_side_info['depth'], veh_side_info['depth']


    dataset, opt, pipe = create_params(args)
    testing_iterations = args.test_iterations
    saving_iterations = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations
    checkpoint = args.start_checkpoint
    debug_from = args.debug_from
    usedepth = args.depth
    usedepthReg = args.usedepthReg

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians_inf = GaussianModel(dataset.sh_degree)
    gaussians_veh = GaussianModel(dataset.sh_degree)
    # TODO: implement a new class for multi gaussian splatting

    # TODO: ↓ load cameras
    dair_info = sceneLoadTypeCallbacks['V2X'](dair_item)  # lidar coordinate -> world coordinate
    inf_scene = Scene(
        model_path = dair_item.model_path, dair_info = dair_info, gaussians = gaussians_inf,
        side_info = inf_side_info, type = 'inf'
    )
    veh_scene = Scene(
        model_path=dair_item.model_path, dair_info=dair_info, gaussians=gaussians_veh,
        side_info=veh_side_info, type='veh'
    )
    # TODO: transport depth&v2x-scene here
    gaussians_inf.training_setup(opt)
    gaussians_veh.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians_inf.restore(model_params, opt)
        gaussians_veh.restore(model_params, opt)
        # TODO: implement a class that fit a series of 3DGS

    # Temporary using 2 3DGS
    omit = torch.tensor(0.5).to('cuda').reshape(1)
    omit = torch.nn.Parameter(omit.detach().requires_grad_(True))
    # 变成一个带权的高斯，用来操控 重叠/空缺 部分的深度信息 -> 一个scalar可控的，用来组合场景的中间件
    # 拓展：能以可微的方式优化路端视角 -> 在真实场景中，路端视角和车端视角之间的具体变换关系可能不知道？

    inf_fg_mask, veh_fg_mask = inf_side_info['mask'], veh_side_info['mask']
    torch.empty_cache()

    # TODO-1: find where to read camera intrinsics/extrinsics, amend them respectively.
    # TODO-2: original dataset pre stored multi-views, while we only sample two views.
    # TODO-3: improve depth rasterization, depth & mask optimization

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # why background

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
                    """
                        这里其实有问题，如果原来对点云做了合并，最终渲染得到的图片应该是更大（或一样）的，但是这里的图片肯定不是
                    """
                    net_image_inf = render(custom_cam, gaussians_inf, pipe, background, scaling_modifer)["render"]
                    net_image_veh = render(custom_cam, gaussians_veh, pipe, background, scaling_modifer)["render"]
                    print(f"[Debug] | net_image_inf.shape = {net_image_inf.shape}, net_image_veh.shape = {net_image_veh.shape}, omit.shape = {omit.shape}")

                    # 这里不能简单地做叠加
                    net_image = net_image_inf * omit + net_image_veh * (1. - omit)
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians_inf.update_learning_rate(iteration)
        gaussians_veh.update_learning_raee(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            if not (usedepth and iteration >= 2000):
                gaussians_inf.oneupSHdegree()
                gaussians_veh.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # depth存在render_pkg里，如果有其他要计算的也可以封装到这里，render_pkg['depth']访问深度
        render_pkg_inf = render(viewpoint_cam, gaussians_inf, pipe, bg)
        render_pkg_veh = render(viewpoint_cam, gaussians_veh, pipe, bg)

        # weighted
        image_inf, image_veh = render_pkg_inf["render"], render_pkg_veh["render"]
        # image = render_pkg_inf["render"] * omit + render_pkg_veh["render"] * (1. - omit)

        # viewspace_point_tensor = torch.zeros_like(
        #     torch.cat([gaussians_inf.get_xyz, gaussians_veh.get_xyz], dim=1), dtype=gaussians_inf.get_xyz.dtype, requires_grad=True, device="cuda"
        # ) + 0

        # TODO: Bind
        # 如何合并？radii, visibility_filter都是用来控制GS球分裂&合并的，Densification
        visibility_filter_inf, radii_inf, viewspace_point_tensor_inf = render_pkg_inf["visibility_filter"], render_pkg_inf["radii"], render_pkg_inf['viewspace_points']
        visibility_filter_veh, radii_veh, viewspace_point_tensor_veh = render_pkg_veh["visibility_filter"], render_pkg_veh["radii"], render_pkg_veh['viewspace_points']


        # Loss
        # gt_image = viewpoint_cam.original_image.cuda()
        # TODO: rgb的损失要分别传给对应的GS, depth是一起渲染的, 一起传播 | 修改上面这行代码，以及对应到的class下的成员读取
        gt_image_inf, gt_image_veh = ...?
        Ll1 = l1_loss(image_inf, gt_image_inf) * omit + l1_loss(image_veh, gt_image_veh) * (1. - omit)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * \
               (1.0 - ssim(image_inf, gt_image_inf) * omit - ssim(image_veh, gt_image_veh) * (1. - omit))


        ### depth supervised loss
        depth_inf, depth_veh = render_pkg_inf["depth"], render_pkg_veh["depth"]
        # if usedepth and viewpoint_cam.original_depth is not None:
        if usedepth:
            # depth_mask = inf_fg_mask if gaussians is gaussians_inf else veh_fg_mask  # render_pkg["acc"][0]
            gt_maskeddepth_inf_fg = (gt_depth_inf_dict['fg'] * inf_fg_mask).cuda()
            gt_maskeddepth_inf_bg = (gt_depth_inf_dict['bg'] * (1.-inf_fg_mask)).cuda()
            gt_maskeddepth_veh_fg = (gt_depth_veh_dict['fg'] * veh_fg_mask).cuda()
            gt_maskeddepth_veh_bg = (gt_depth_veh_dict['dg'] * (1.-veh_fg_mask)).cuda()
            if args.white_background:  # for 360 datasets ...
                gt_maskeddepth_inf_fg = normalize_depth(gt_maskeddepth_inf_fg)
                gt_maskeddepth_inf_bg = normalize_depth(gt_maskeddepth_inf_bg)
                gt_maskeddepth_veh_fg = normalize_depth(gt_maskeddepth_veh_fg)
                gt_maskeddepth_veh_bg = normalize_depth(gt_maskeddepth_veh_bg)

                depth_inf = normalize_depth(depth_inf)
                depth_veh = normalize_depth(depth_veh)

            deploss = l1_loss(gt_maskeddepth_inf_fg, depth_inf * inf_fg_mask) * 0.5 + \
                      l1_loss(gt_maskeddepth_inf_bg, depth_inf * (1. - inf_fg_mask)) * 0.5 + \
                      l1_loss(gt_maskeddepth_veh_fg, depth_veh * veh_fg_mask) * 0.5 + \
                      l1_loss(gt_maskeddepth_veh_bg, depth_veh * (1. - veh_fg_mask)) * 0.5

            loss = loss + deploss

        ## depth regularization loss (canny)
        if usedepthReg and iteration >= 0:

            depth_mask_inf, depth_mask_veh = (depth_inf > 0).detach(), (depth_veh > 0).detach()
            # Ori Name: nearDepthMean_map
            map_inf = nearMean_map(depth_inf, viewpoint_cam.canny_mask * depth_mask_inf, kernelsize=3)
            map_veh = nearMean_map(depth_veh, viewpoint_cam.canny_mask * depth_mask_veh, kernelsize=3)
            loss = loss + l2_loss(map_inf, depth_inf * depth_mask_inf) * 1.0 + l2_loss(map_veh, depth_veh * depth_mask_veh) * 1.0



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
                    Reporter(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
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
            Reporter(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background), txt_path=os.path.join(args.model_path, "metric.txt"))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # TODO: Respectively
            for gaussian_item in [
                (gaussians_inf, visibility_filter_inf, radii_inf, viewspace_point_tensor_inf, 'inf'),
                (gaussians_veh, visibility_filter_veh, radii_veh, viewspace_point_tensor_veh, 'veh')
            ]:
                gaussians, visibility_filter, radii, viewspace_point_tensor, data_type = gaussian_item
            # Densification
                if iteration < opt.densify_until_iter and ((not usedepth) or gaussians._xyz.shape[0] <= 1500000):
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # camera_extent ?
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
                    torch.save((gaussians.capture(), iteration), scene.model_path + f"/{data_type}-chkpnt" + str(iteration) + ".pth")

    pass



def parser_add(parser=None):
    if parser is None:
        parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[30_000])  # default=([1, 250, 500,]+ [i*1000 for i in range(1,31)]))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--usedepthReg", action="store_true")

    return parser




def main():
    base_dir = '../'
    dair = DAIR_V2X_C(base_dir)
    from random import randint
    prepared_idx = randint(0, 1000) % 600  # random
    pair = CooperativeData(dair[prepared_idx], base_dir) # dair_item

    processed_dict = process_first(parser = None,dair_item = pair)
    """
    {
        'inf-side': pred_depth[0],
        'veh-side': pred_depth[1],
        'args': opt,
        'parser': parser
    }
    
    
    xx-side ->
        pred_depth[i]: 
            {
                'depth': {
                'fg': (colored_pred_fg, colored_init_fg, pred_fg),      -> np.array - [H W 3]
                'bg': (colored_pred_bg, colored_init_bg, pred_bg),      -> np.array - [H W 3]
                'panoptic': (colored_pred_all, colored_init, pred)      -> np.array - [H W 3]
                }, 'pcd': pcd_file # use open3d.io.read_point_cloud(...)
            }
    
    -> pred_fg, pred_bg, pred -> uncolored
    
    """
    parser = processed_dict['parser']
    # {'depth': ..., 'pcd': ...}
    inf_side = processed_dict['inf-side']
    veh_side = processed_dict['veh-side']


    parser = parser_add(parser)
    train_DRGS(
        args = parser.parse_args(), dair_item = pair,
        inf_side_info = inf_side, veh_side_info = veh_side
    )






if __name__ == "__main__":
    main()
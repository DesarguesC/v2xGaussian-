import pdb
from einops import rearrange
from process import process_first
import os, sys, uuid, cv2, torch, torchvision
from tqdm import tqdm
import open3d as o3d
from random import randint
from basicsr.utils import tensor2img, img2tensor

from drgs_utils.gaussian_renderer import render, network_gui
from drgs_utils.scene import GaussianModel
from drgs_utils.utils.loss_utils import l1_loss, l2_loss, nearMean_map, ssim
from drgs_utils.utils.image_utils import normalize_depth, psnr
from drgs_utils.lpipsPyTorch import lpips
from drgs_utils import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    depth_colorize_with_mask
)

from drgs_utils.scene import Scene, sceneLoadTypeCallbacks, sceneConbinationCallbacks
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





def train_DRGS(
        args, dair_item: CooperativeData,
        inf_side_info: dict, veh_side_info:dict,
        inf_view_veh: np.array, veh_view_inf: np.array
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

    # TODO: create params
    lp = ModelParams(args)
    op = OptimizationParams(args)
    pp = PipelineParams(args)
    args = args.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    w, h = inf_side_info['rgb'].size
    if not hasattr(args, 'w'): setattr(args, 'w', w)
    else: args.w = w
    if not hasattr(args, 'h'): setattr(args, 'h', h)
    else: args.h = h
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)


    # TODO: cut down points in pcd
    inf_side_info['pcd'] = cut_down_points(inf_side_info['pcd'], 1. / args.downsample)
    veh_side_info['pcd'] = cut_down_points(veh_side_info['pcd'], 1. / args.downsample)

    testing_iterations = args.test_iterations
    # saving_iterations = args.save_iterations # [30000, 30000] ?
    saving_iterations = [29, 299, 2990, 29990] # TEST
    checkpoint_iterations = args.checkpoint_iterations
    checkpoint = args.start_checkpoint
    # debug_from = args.debug_from
    usedepth = args.depth
    usedepthReg = args.usedepthReg

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians_inf = GaussianModel(dataset.sh_degree)
    gaussians_veh = GaussianModel(dataset.sh_degree)
    # TODO: implement a new class for multi gaussian splatting

    # TODO: ↓ load cameras
    dair_info = sceneLoadTypeCallbacks['V2X'](dair_item, inf_side_info, veh_side_info)  # lidar coordinate -> world coordinate

    inf_scene = Scene(
        args = dataset, dair_item = dair_item, dair_info = dair_info,
        gaussians = gaussians_inf, side_info = inf_side_info, type = 'inf'
    )
    veh_scene = Scene(
        args = dataset, dair_item = dair_item, dair_info=dair_info,
        gaussians=gaussians_veh, side_info=veh_side_info, type='veh'
    )


    # 在世界坐标系下合并点云, 使用inf_view_veh和veh_view_inf
    # Train_Scene = sceneConbinationCallbacks['conbine-pcd'](inf_scene, veh_scene)
    # TODO: train_camera_list & test_camera_list
    # TODO: 这里只能合并点云, 两个scene还是得分开优化, 因为有两个高斯; 前面train_camera和test_camera里有随机索引, 我干脆每次随机挑一个优化
    # -> train_camera/test_camera 另外实现一个类, 调到哪个就优化哪个类

    # TODO: transport depth&v2x-scene here
    gaussians_inf.training_setup(opt)
    gaussians_veh.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians_inf.restore(model_params, opt)
        gaussians_veh.restore(model_params, opt)
        # TODO: implement a class that fit a series of 3DGS

    # Temporary using 2 3DGS
    # omit = torch.tensor(0.5).to('cuda').reshape(1)
    # 变成一个带权的高斯，用来操控 重叠/空缺 部分的深度信息 -> 一个scalar可控的，用来组合场景的中间件
    # 拓展：能以可微的方式优化路端视角 -> 在真实场景中，路端视角和车端视角之间的具体变换关系可能不知道？


    meta = torch.zeros((args.h, args.w)).to('cuda')
    assert args.h%8==0 and args.w%8==0, f'(args.w, args.h) = {(args.w, args.h)}'
    h_, w_ = args.h//8, args.w//8
    meta_valid = torch.ones((h_*6, w_*6)).to('cuda')
    meta[h_:args.h-h_, w_:args.w-w_] = meta_valid

    # Calculate a mask ?
    with torch.no_grad():
        foc1_a = torch.nn.Parameter((0.5 * meta).clone().detach().requires_grad_(True)).cuda()
        foc1_b = torch.nn.Parameter((0.5 * meta).clone().detach().requires_grad_(True)).cuda()
        foc2_a = torch.nn.Parameter((0.5 * meta).clone().detach().requires_grad_(True)).cuda()
        foc2_b = torch.nn.Parameter((0.5 * meta).clone().detach().requires_grad_(True)).cuda()


    torch.cuda.empty_cache()

    # TODO-1: find where to read camera intrinsics/extrinsics, amend them respectively.
    # TODO-2: original dataset pre stored multi-views, while we only sample two views.
    # TODO-3: improve depth rasterization, depth & mask optimization

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # why background

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    TrainTargets = [
        # 0 -> infrastructure side
        {
            'gaussian': gaussians_inf,
            'scene': inf_scene,
            'mask': inf_side_info['mask'], # fg-mask
            'depth': inf_side_info['depth'],
            'view': inf_view_veh,  # view vehicle pcd from infrastructure side -> update
            'foc': [foc1_a, foc1_b],
            'name': 'inf'
        },
        # 1 -> vehicle side
        {
            'gaussian': gaussians_veh,
            'scene': veh_scene,
            'mask': veh_side_info['mask'], # fg-mask
            'depth': veh_side_info['depth'],
            'view': veh_view_inf,  # view infrastructure pcd from vehicle side -> update
            'foc': [foc2_a, foc2_b],
            'name': 'veh'
        }
    ]

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_depthloss_for_log, prev_depthloss, deploss = 0.0, 1e2, torch.zeros(1)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    pdb.set_trace()
    try:

        for iteration in range(first_iter, opt.iterations + 1):
            # if iteration >= 3099:
            #     pdb.set_trace()

            train_now_idx = randint(0,1) # [0,1]
            train_now = TrainTargets[train_now_idx]

            gaussian = train_now['gaussian']
            scene = train_now['scene']

            if 0 in scene.gaussians._xyz.shape:
                opt.iterations += 1
                continue

            fg_mask = rearrange(torch.tensor(train_now['mask']), 'h w c -> c h w').requires_grad_(False).cuda()

            with torch.no_grad():
                bg_mask = 1. - fg_mask
            dep = rearrange(torch.tensor(train_now['depth']['panoptic'][-1]), 'h w c -> c h w').requires_grad_(False).cuda() # panoptic, uncolored
            viewer_depth = torch.tensor(train_now['view']).requires_grad_(False).cuda()

            foc = train_now['foc']
            extra_name = train_now['name']


            # 当前结果实时渲染
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
                        net_image = render(custom_cam, gaussian, pipe, background, scaling_modifer)["render"]
                        # net_image_inf = render(custom_cam, gaussians_inf, pipe, background, scaling_modifer)["render"]
                        # print(f"[Debug] | net_image.shape = {net_image.shape}, foc.shape = {foc.shape}")
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    print(f'err: {e}')
                    network_gui.conn = None

            iter_start.record()
            gaussian.update_learning_rate(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                if not (usedepth and iteration >= 2000):
                    gaussian.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1)) # 弹出任意位置的元素(并删除)，非严格的栈结构
            # TODO: 原始DRGS代码再viewpoint_cam.original_depth和viewpoint_cam.original_iamge处提供了ground truth，我换个地方提供

            # Render
            # if (iteration - 1) == debug_from:
            # TODO: debug
            pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            # depth存在render_pkg里，如果有其他要计算的也可以封装到这里，render_pkg['depth']访问深度

            render_pkg = render(viewpoint, gaussian, pipe, bg)
            image_side_rendered, depth_rendered = render_pkg["render"], render_pkg['depth']

            # pdb.set_trace()  # Double Check: viewpoint
            depth_rendered = foc[0] * depth_rendered + foc[1] * viewer_depth
            depth_rendered = normalize_depth(depth_rendered)
            # foc * depth_rendered ? foc * viewer_depth ?
            # TODO: Bind
            # 如何合并？radii, visibility_filter都是用来控制GS球分裂&合并的，Densification
            visibility_filter, radii, viewspace_point_tensor = render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['viewspace_points']

            # Loss
            # gt_image = viewpoint_cam.original_image.cuda()
            # TODO: rgb的损失要分别传给对应的GS, depth是一起渲染的, 一起传播 | 修改上面这行代码，以及对应到的class下的成员读取
            gt_image, gt_depth = viewpoint.original_image, viewpoint.original_depth
            # gt_image_inf, gt_image_veh = ...?
            Ll1 = l1_loss(gt_image, image_side_rendered)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                       1.0 - ssim(image_side_rendered * fg_mask, gt_image * fg_mask) \
                       - ssim(image_side_rendered * bg_mask, gt_image * bg_mask)
                )


            # print(f'[Debug] depth_rendered.shape = {depth_rendered.shape}, fg_mask.shape = {fg_mask.shape}')
            deploss = 0.5 * l2_loss((dep * fg_mask).cuda(), (depth_rendered * fg_mask).cuda()) + 0.5 * l2_loss((dep * bg_mask).cuda(), (depth_rendered * bg_mask).cuda())
            loss = loss + deploss

            # pdb.set_trace()
            ## depth regularization loss (canny)
            if usedepthReg and iteration >= 0:
                depth_mask = (depth_rendered > 0).detach()
                # depth_mask_inf, depth_mask_veh = (depth_inf > 0).detach(), (depth_veh > 0).detach()
                # Ori Name: nearDepthMean_map
                depth_map = nearMean_map(depth_mask, viewpoint.canny_mask * depth_mask, kernelsize=3)
                # map_veh = nearMean_map(depth_veh, viewpoint.canny_mask * depth_mask_veh, kernelsize=3)
                loss = loss + l2_loss(depth_map, depth_rendered * depth_mask) * 1.0 + l2_loss(depth_map, depth_rendered * depth_mask) * 1.0

            loss.backward()
            iter_end.record()


            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_depthloss_for_log = 0.2 * deploss.item() + 0.8 * ema_depthloss_for_log
                if iteration % 10 == 0:
                    # pdb.set_trace()
                    progress_bar.set_postfix(
                        {"Loss": f"{ema_loss_for_log:.{7}f}", "Deploss": f"{ema_depthloss_for_log:.4f}",
                         "#pts": gaussian._xyz.shape[0]})
                    progress_bar.update(10)

                if iteration % 10000 == 0:
                    # if iteration % 1000 == 0:
                    #     pdb.set_trace()
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
                    # print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter and ((not usedepth) or gaussian._xyz.shape[0] <= 1500000):
                    # Keep track of max radii in image-space for pruning

                    # pdb.set_trace()

                    gaussian.max_radii2D[visibility_filter] = torch.max(gaussian.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # camera_extent ?
                        gaussian.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold)

                    if not usedepth:
                        if iteration % opt.opacity_reset_interval == 0 or (
                                dataset.white_background and iteration == opt.densify_from_iter):
                            gaussian.reset_opacity()


                    # Optimizer step
                    if iteration < opt.iterations:
                        with torch.no_grad():
                            gaussian.optimizer.step()
                            gaussian.optimizer.zero_grad(set_to_none=True)

                    if (iteration in checkpoint_iterations):
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((gaussian.capture(), iteration), scene.model_path + f"/{extra_name}-chkpnt" + str(iteration) + ".pth")

    except Exception as err:
        print(f'[Debug] Err: {err}')
        pdb.set_trace()


    print('\nTraining Process Finished.\n')
    # save weights: foc[i,j], depth, camera-param

    try:
        M1_path, M2_path = os.path.join(args.save_dir, 'M1'), os.path.join(args.save_dir, 'M2')
        if not os.path.exists(M1_path): os.mkdir(M1_path)
        if not os.path.exists(M2_path): os.mkdir(M2_path)
        torch.save(foc1_a, os.path.join(args.save_dir, 'foc1_a.pth')) # 直接torch.load即可
        torch.save(foc1_b, os.path.join(args.save_dir, 'foc1_b.pth'))
        torch.save(foc2_a, os.path.join(args.save_dir, 'foc2_a.pth'))
        torch.save(foc2_b, os.path.join(args.save_dir, 'foc2_b.pth'))

        # if not viewpoint_stack:
        viewpoint_stack_inf, viewpoint_stack_veh = inf_scene.getTrainCameras().copy(), veh_scene.getTrainCameras().copy()
        viewpoint_inf, viewpoint_veh = viewpoint_stack_inf.pop(randint(0, len(viewpoint_stack_inf) - 1)), viewpoint_stack_veh.pop(randint(0, len(viewpoint_stack_veh) - 1))


        bg = torch.rand((3), device="cuda") if opt.random_background else background
        # depth存在render_pkg里，如果有其他要计算的也可以封装到这里，render_pkg['depth']访问深度

        render_pkg = render(viewpoint_inf, gaussians_inf, pipe, bg)
        img_inf = tensor2img(render_pkg["render"])
        depth_rendered = normalize_depth(foc1_a * render_pkg['depth'] + foc1_b * torch.tensor(inf_view_veh, device=foc1_b.device))
        dep_inf = tensor2img(depth_rendered)

        print(f'img_inf.shape = {img_inf.shape}, dep_inf.shape = {dep_inf.shape}')
        cv2.imwrite(os.path.join(M1_path, 'rgb.jpg'), img_inf)
        cv2.imwrite(os.path.join(M1_path, 'dep.jpg'), dep_inf)


        render_pkg = render(viewpoint_veh, gaussians_veh, pipe, bg)
        depth_rendered = normalize_depth(foc2_a * render_pkg['depth'] + foc2_b * torch.tensor(veh_view_inf, device=foc2_b.device))
        img_veh = tensor2img(render_pkg["render"])
        dep_veh = tensor2img(depth_rendered)

        print(f'img_veh.shape = {img_veh.shape}, dep_veh.shape = {dep_veh.shape}')
        cv2.imwrite(os.path.join(M2_path, 'rgb.jpg'), img_veh)
        cv2.imwrite(os.path.join(M2_path, 'dep.jpg'), dep_veh)

    except Exception as err:
        print(f'err: {err}')
        pdb.set_trace()






    pdb.set_trace()



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

    base_dir = '../dair-test'
    save_dir = os.path.join(base_dir, 'weights')
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    dair = DAIR_V2X_C(base_dir)
    from random import randint
    # prepared_idx = randint(0, 1000) % 600  # random
    prepared_idx = 0 # TEST
    pair = CooperativeData(dair[prepared_idx], base_dir) # dair_item

    read_only = bool(int(os.environ.get('READ_ONLY')))
    print(f'READ_ONLY = {read_only}')
    # exit(0)
    processed_dict = process_first(parser = None, dair_item = pair, debug_part = False, read_only=read_only)
    print('-'*20 + 'Finish Reading' + '-'*20)
    """
    {
        'inf-side': pred_depth[0],
        'veh-side': pred_depth[1],
        'args': opt,
        'parser': parser
        'inf-side-veh': pred_depth[2]['side-depth'], # mapping depth from veh-pcd to infrastructure view
        'veh-side-inf': pred_depth[3]['side-depth'], # mapping depth from inf-pcd to vehicle view
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
    # pair.set_downsample(parser.downsample)
    parser = processed_dict['parser']
    parser.debug_mode = True # manually set [temporary]

    # {'depth': ..., 'pcd': ...}
    inf_side = processed_dict['inf-side']
    veh_side = processed_dict['veh-side']
    inf_view_veh = processed_dict['inf-side-veh']
    veh_view_inf = processed_dict['veh-side-inf']

    parser = parser_add(parser)
    parser.add_argument('--save_dir', type=str, default=save_dir)
    # save foc[i,j] here, save depth/camera-param here


    train_DRGS(
        args = parser, dair_item = pair,
        inf_side_info = inf_side, veh_side_info = veh_side,
        inf_view_veh = inf_view_veh, veh_view_inf = veh_view_inf
    )

def cut_down_points(pcd, pro: float):
    # pdb.set_trace()
    # pro: 1. / opt.downsample
    x = np.asarray(pcd.points)
    uu = np.concatenate([x[i].reshape((1, 3)) for i in range(x.shape[0]) if (randint(0, 999) < 1e3 * pro)], axis=0)
    pcd.points = o3d.cuda.pybind.utility.Vector3dVector(uu)
    return pcd





if __name__ == "__main__":
    main()
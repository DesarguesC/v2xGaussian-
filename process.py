import os, glob, cv2, argparse, torch, rembg, sys, pdb
import numpy as np
import open3d as o3d
from PIL import Image
from cam_utils import downsampler
from lidar2dep.config import Get_Merged_Args, get_args_parser
from lidar2dep.main import Args2Results, Direct_Renderring
from seem.utils.constants import COCO_PANOPTIC_CLASSES
from seem.masks import FG_remove, FG_remove_All, preload_seem_detector, preload_lama_remover
from lidar2dep.dair import DAIR_V2X_C, CooperativeData

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def wirte_pred_depth(pred_item: dict = None, idx: int = None):
    """
    {
                'rgb': image, # pil
                'mask': mask, # fg-mask
                'depth': {
                'fg': (colored_pred_fg, colored_init_fg, pred_fg),
                'bg': (colored_pred_bg, colored_init_bg, pred_bg),
                'panoptic': (colored_pred_all, colored_init, pred) # TODO: adjust it onto PointCloud again ?
                }, 'pcd': pcd_file
    }
    """
    dd = './debug'


    if 'side-pred' in pred_item:
        pic_path = os.path.join(dd, f'side-pred-{idx}')
        if not os.path.exists(pic_path): os.mkdir(pic_path)
        cv2.imwrite(pred_item['side-depth'], os.path.join(pic_path, f'{idx:03}.jpg'))
    else:
        dd = os.path.join(dd, f'pred_dep_{idx}')
        if not os.path.exists(dd): os.mkdir(dd)
        rgb_path = os.path.join(dd, f'rgb.jpg')
        mask_path = os.path.join(dd, 'fg_mask.jpg')
        depth_path = os.path.join(dd, 'depth_image') # only panoptic images
        if not os.path.exists(depth_path): os.mkdir(depth_path)

        pred_item['rgb'].save(rgb_path)
        cv2.imwrite(pred_item['mask'], mask_path)
        named = ['colored_pred_all', 'colored_init', 'pred']
        for i in range(3):
            cv2.imwrite(pred_item['depth']['panoptic'][i], os.path.join(depth_path, named[i]))








def process_first(
        parser = None, dair_item: CooperativeData = None
        # rgb_file_path: list[str] = None, pcd_file_path: list[str] = None,
        # intrinsic_path: list[str] = None, extrinsic_path: list[str] = None
):
    # assert dair_item is not None
    if parser is None:
        parser = argparse.ArgumentParser()

    # parser.add_argument('--path', default="./data/test1.jpg", type=str, help="path to image (png, jpeg, etc.)")
    # parser.add_argument('--model', default='u2net', type=str, help="rembg model, see https://github.com/danielgatis/rembg#models")
    parser.add_argument('--size', default=256, type=int, help="output resolution")
    parser.add_argument('--downsample', default=4, type=int, help="downsample scale")
    parser.add_argument('--border_ratio', default=0.2, type=float, help="output border ratio")
    parser.add_argument('--recenter', type=bool, default=True, help="recenter, potentially not helpful for multiview zero123")
    # SEEM
    parser.add_argument('--seem_ckpt', type=str, default="../Tools/SEEM/seem_focall_v0.pt", help='restore where the SEEM & LaMa model locates')
    parser.add_argument('--seem_cfg', type=str, default="seem/configs/seem/focall_unicl_lang_demo.yaml")
    # LaMa
    parser.add_argument('--lama_ckpt', type=str, default='../Tools/LaMa/', help='actually path to lama ckpt base folder, ckpt specified in config files')
    parser.add_argument('--lama_cfg', type=str, default='./configs/lama_default.yaml', help='path to lama inpainting config path')
    # LLM
    parser.add_argument('--use_llm', type=str2bool, default=False, help='whether to use Claude or not')
    #Outputs
    parser.add_argument('--results', type=str, default='../v2x-outputs/pre-process/', help='result direction')
    print(f'parser = {parser}')
    opt = get_args_parser(parser=parser)
    dair_item.set_downsample(opt.downsample)

    # if dair_item is not None:
    #     opt.rgb_file_path = rgb_file_path
    #     opt.pcd_file_path = pcd_file_path
    #     opt.intrinsic_path = intrinsic_path
    #     opt.extrinsic_path = extrinsic_path

    print('Start...')
    """
        create results directions here ↓
        MASK: os.path.join(opt.results, 'masks')
        VIEW: os.path.join(opt.results,  'views')
    """
    if not os.path.exists(opt.results):
        os.mkdir(opt.results)
    if not os.path.exists(os.path.join(opt.results, 'remove')):
        os.mkdir(os.path.join(opt.results, 'remove'))

    setattr(opt, "device", "cuda" if torch.cuda.is_available() else "cpu")

    # session = rembg.new_session(model_name=opt.model)

    # if os.path.isdir(opt.rgb_file_path):
    #     print(f'[INFO] processing directory {opt.rgb_file_path}...')
    #     files = glob.glob(f'{opt.rgb_file_path}/*')
    #     # out_dir = opt.rgb_file_path
    # else: # isfile
    #     files = [opt.rgb_file_path]
    #     # out_dir = os.path.dirname(opt.results)


    preloaded_seem_detector = preload_seem_detector(opt)
    preloaded_lama_dict = preload_lama_remover(opt)

    # pdb.set_trace()

    files = [
        {
            'rgb': dair_item.inf_img_path, 'pcd': dair_item.inf_pcd_path,
            'camera': dair_item.load4pcd_render(type='inf'), 'extra': 'inf'
         },
        {
            'rgb': dair_item.veh_img_path, 'pcd': dair_item.veh_pcd_path,
            'camera': dair_item.load4pcd_render(type='veh'), 'extra': 'veh'
        },
        # mapping depth from veh-pcd to infrastructure view
        {
            'rgb': None, 'pcd': dair_item.inf_pcd_path,
            'camera': dair_item.load4pcd_render(type='veh'), 'extra': 'inf-side-veh'
        },
        # mapping depth from inf-pcd to vehicle view
        {
            'rgb': None, 'pcd': dair_item.veh_pcd_path,
            'camera': dair_item.load4pcd_render(type='inf'), 'extra': 'veh-side-inf'
        }
    ]

    pred_depth = []

    print(f'files: {files}')
    for file in files:
        rgb_file = file['rgb'] # pil
        pcd_file_path = file['pcd']
        pcd_file = o3d.io.read_point_cloud(pcd_file_path)
        camera = file['camera']
        extra_name = file['extra']

        # pdb.set_trace()

        if rgb_file is None:
            # 算变换矩阵
            if 'inf' in extra_name: # inf view veh: veh_lidar 2 world 2 inf_lidar 2 inf_cam
                veh_lidar2world = dair_item.veh_side_pcd2world()
                world2inf_cam = dair_item.world2inf_cam()
                cam_extrinsics = veh_lidar2world @ world2inf_cam

            else:
                inf_lidar2world = dair_item.inf_side_pcd2world()
                world2veh_cam = dair_item.world2veh_cam()
                cam_extrinsics = inf_lidar2world @ world2veh_cam

            direct_result = Direct_Renderring(pcd_file, opt.depth_path, extra_name, camera, cam_extrinsics)
            # {"side-depth": depth}
            pred_depth.append(direct_result)
            continue

        # pdb.set_trace()

        # load image
        print(f'[INFO] loading image {rgb_file}...')
        image = Image.open(rgb_file) # RGB Image
        image = downsampler(image, opt.downsample)

        # TODO: use seem to remove foreground
        print(f'[INFO] background removal...')
        res, mask, carved_image = FG_remove_All(
            opt = opt, img = image,
            preloaded_seem_detector=preloaded_seem_detector, preloaded_lama_dict=preloaded_lama_dict,
            use_llm=opt.use_llm
        )

        _, _, carved_image_fg = FG_remove_All(
            opt=opt, img=image, mask = 1.-mask,
            preloaded_seem_detector=preloaded_seem_detector, preloaded_lama_dict=preloaded_lama_dict,
            use_llm=opt.use_llm
        )

        mask, res, carved_image, carved_image_fg = np.uint8(mask), np.uint8(res), np.uint8(carved_image), np.uint8(carved_image_fg)
        # TODO: save intermediate results
        cv2.imwrite(os.path.join(opt.results, f'remove/mask-{extra_name}.jpg'), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(opt.results, f'remove/res-{extra_name}.jpg'), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(opt.results, f'remove/removed-{extra_name}-bg.jpg'), cv2.cvtColor(carved_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(opt.results, f'remove/removed-{extra_name}-fg.jpg'), cv2.cvtColor(carved_image_fg, cv2.COLOR_RGB2BGR))

        pdb.set_trace()

        # BackGround
        colored_pred_bg, colored_init_bg, pred_bg = \
            Args2Results(
                opt, rgb_file=carved_image, pcd_file_path=pcd_file,
                intrinsics=camera['intrinsic'], extrinsics=camera['extrinsic'],
                fg_mask=mask, new_path=False, extra_name=f'{extra_name}-bg'
            )
        # ForeGround
        # colored_pred_fg, colored_init_fg, pred_fg = \
        #   Args2Results(opt, rgb_file=np.array(image)*mask, fg_mask=1.-mask, new_path=False, extra_name='fg')
        #   前景没有背景
        colored_pred_fg, colored_init_fg, pred_fg = \
            Args2Results(
                opt, rgb_file=carved_image_fg, pcd_file_path=pcd_file,
                intrinsics=camera['intrinsic'], extrinsics=camera['extrinsic'],
                fg_mask=1.-mask, new_path=False, extra_name=f'{extra_name}-fg'
            )
        #   前景使用lama填充背景

        # 全景用img+pcd估计深度后，前背景分离监督
        colored_pred_all, colored_init, pred = \
            Args2Results(
                opt, rgb_file=np.array(image), pcd_file_path=pcd_file,
                intrinsics=camera['intrinsic'], extrinsics=camera['extrinsic'],
                fg_mask=None, new_path=False, extra_name=f'{extra_name}-panoptic'
            )
        # 不分前背景
        print(colored_pred_bg, colored_pred_fg)
        print(colored_init_bg, colored_init_fg)
        print(pred_bg, pred_fg)
        # np.array - [H W 3]

        pdb.set_trace()

        pred_depth.append({
                'rgb': image, # pil
                'mask': mask, # fg-mask
                'depth': {
                'fg': (colored_pred_fg, colored_init_fg, pred_fg),
                'bg': (colored_pred_bg, colored_init_bg, pred_bg),
                'panoptic': (colored_pred_all, colored_init, pred) # TODO: adjust it onto PointCloud again ?
                }, 'pcd': pcd_file
            })

    print('\nDone.')


    assert len(pred_depth) == 4, f'len(pred_depth) = {len(pred_depth)}'

    for i in range(len(pred_depth)):
        wirte_pred_depth(pred_depth[i], i)

    return {
        'inf-side': pred_depth[0],
        'veh-side': pred_depth[1],
        'args': opt,
        'parser': parser,
        'inf-side-veh': pred_depth[2]['side-depth'], # mapping depth from veh-pcd to infrastructure view
        'veh-side-inf': pred_depth[3]['side-depth'], # mapping depth from inf-pcd to vehicle view
    }




if __name__ == '__main__':
    base_dir = '../dair-test'
    dair = DAIR_V2X_C(base_dir)
    from random import randint

    # prepared_idx = randint(0, 1000) % 600  # random
    prepared_idx = 0 # TEST
    pair = CooperativeData(dair[prepared_idx], base_dir)  # dair_item

    processed_dict = process_first(parser=None, dair_item=pair)

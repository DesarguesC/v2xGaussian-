import os, pdb
import json
import hydra
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF
from flash3d.models.model import GaussianPredictor, to_device
from evaluation.evaluator import Evaluator
from flash3d.datasets.util import create_datasets # stuck ?
from misc.visualise_3d import save_ply
from flash3d.datasets.infer import InferenceV2X

from misc.util import add_source_frame_id


def get_model_instance(model):
    """
    unwraps model from EMA object
    """
    return model.ema_model if type(model).__name__ == "EMA" else model

# def operator(input_image: np.ndarray) -> dict:
#     # input_image = cv2.imread(...) or Image.open(...)
#     # TODO: how do I transfer the input_image to a correct type ?
#
#     # see: models.encoder.unidepth_encoder.py -> class: UniDepthExtended.forward
#     # see: models.model.py -> class: GaussianPredictor.compute_gauss_means ❎ 不用看，内部自己操作的
#
#     """
#         inputs[('unidepth', 0, 0)]
#         inputs[("K_src", 0)]
#         inputs["color_aug", 0, 0]
#
#         TODO: 看在KITTI上的实现
#         construction method: see "__getitem__" of the class in datasets/{re10k.py, dataset.py, kitti.py}
#     """
#
#     target_dict = {
#         'target_frame_ids': ...,
#
#     }
#
#
#
#     return input_image


def evaluate(opt, model, cfg, evaluator, dair_info, split='test', view_type='inf', device=None, save_vis=False, return_GS=False):
    out_base_dir = os.path.join(opt.save_dir, 'flash3d')

    out_dir_ply, out_pred_dir = os.path.join(out_base_dir, 'ply'), os.path.join(out_base_dir, 'pred') # create via opt
    out_gt_dir = os.path.join(out_base_dir, 'gt') # create via opt

    for u in [opt.save_dir, out_base_dir, out_dir_ply, out_pred_dir, out_gt_dir]:
        if not os.path.exists(u): os.mkdir(u)
    score_dict = {}
    model_model = get_model_instance(model)
    model_model.set_eval()

    eval_frames = ["s0"]
    target_frame_ids = ["s0"]
    all_frames = add_source_frame_id(eval_frames)
    for fid in all_frames:
        score_dict[fid] = {"ssim": [], "psnr": [], "lpips": [], "name": fid}

    # pdb.set_trace()
    inputs = InferenceV2X(split, cfg, dair_info, view_type=view_type)

    with torch.no_grad():
        # if device is not None:
        #     to_device(inputs, device)

        inputs_item = inputs.getInputs(device)
        inputs_item["target_frame_ids"] = target_frame_ids
        outputs = model(inputs_item) # dict
    # pdb.set_trace()
    for f_id in score_dict.keys():
        pred = outputs[('color_gauss', f_id, 0)]
        if cfg.dataset.name == "dtu":
            gt = inputs_item[('color_orig_res', f_id, 0)]
            pred = TF.resize(pred, gt.shape[-2:])
        else:
            gt = inputs_item[('color', f_id, 0)]

        # pdb.set_trace()
        out = evaluator(pred, gt) # should work in for B>1, however be careful of reduction
        if save_vis:
            save_ply(outputs, f"{out_dir_ply}/{f_id}.ply", gaussians_per_pixel=model.cfg.model.gaussians_per_pixel)
            pred = pred[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            gt = gt[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            plt.imsave(f"{out_pred_dir}/{f_id:03}.png", pred)
            plt.imsave(f"{out_gt_dir}/{f_id:03}.png", gt)

        for metric_name, v in out.items():
            score_dict[f_id][metric_name].append(v)

    metric_names = ["psnr", "ssim", "lpips"]
    score_dict_by_name = {}
    # for f_id in score_dict.keys():
    score_dict_by_name[score_dict[f_id]["name"]] = {}
    for metric_name in metric_names:
        # compute mean
        score_dict[f_id][metric_name] = sum(score_dict[f_id][metric_name]) / len(score_dict[f_id][metric_name])
        # original dict has frame ids as integers, for json out dict we want to change them
        # to the meaningful names stored in dict
        score_dict_by_name[score_dict[f_id]["name"]][metric_name] = score_dict[f_id][metric_name]

    for metric in metric_names:
        vals = [score_dict_by_name[f_id][metric]]
        print(f"{metric}:", np.mean(np.array(vals)))

    return (score_dict_by_name, outputs) if return_GS else score_dict_by_name


def v2x_inference(opt, dair_info, cfg: DictConfig, split='test', view_type: str = 'inf', unidepth_model=None, save_result=True, return_GS=True):

    assert view_type in ['inf', 'veh'], view_type

    if not os.path.exists(opt.save_dir): os.mkdir(opt.save_dir)
    output_dir = os.path.join(opt.save_dir, 'flash3d') # temporary output of Flash3D
    if not os.path.exists(output_dir): os.mkdir(output_dir)

    # ori_dir = os.getcwd() # save for standby
    # os.chdir(output_dir)
    print(f"Saving dir: {output_dir}")

    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 1
    # TODO: for GaussianPredictor loading

    pdb.set_trace()
    model = GaussianPredictor(cfg, unidepth_model=unidepth_model)

    device = torch.device("cuda:0")
    model.to(device)
    if (ckpt_dir := model.checkpoint_dir()).exists():
        # resume training
        model.load_model(ckpt_dir, ckpt_ids=0)

    evaluator = Evaluator() # crop_border = True
    evaluator.to(device)

    # split = "test"
    score_dict_by_name, gaussian_outputs = evaluate(opt, model, cfg, evaluator, dair_info, split, view_type=view_type,
                                  device=device, save_vis=save_result, return_GS=return_GS)
    print(json.dumps(score_dict_by_name, indent=4))
    if cfg.dataset.name=="re10k":
        with open("metrics_{}_{}_{}.json".format(cfg.dataset.name, split, cfg.dataset.test_split), "w") as f:
            json.dump(score_dict_by_name, f, indent=4)
    with open("metrics_{}_{}.json".format(cfg.dataset.name, split), "w") as f:
        json.dump(score_dict_by_name, f, indent=4)

    return score_dict_by_name, gaussian_outputs


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    print("current directory:", os.getcwd())
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    os.chdir(output_dir)
    print("Working dir:", output_dir)

    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 1
    model = GaussianPredictor(cfg)
    device = torch.device("cuda:0")
    model.to(device)
    if (ckpt_dir := model.checkpoint_dir()).exists():
        # resume training
        model.load_model(ckpt_dir, ckpt_ids=0)

    evaluator = Evaluator(crop_border=cfg.dataset.crop_border)
    evaluator.to(device)

    split = "test"
    save_vis = cfg.eval.save_vis
    dataset, dataloader = create_datasets(cfg, split=split)
    score_dict_by_name = evaluate(model, cfg, evaluator, dataloader,
                                  device=device, save_vis=save_vis)
    print(json.dumps(score_dict_by_name, indent=4))
    if cfg.dataset.name == "re10k":
        with open("metrics_{}_{}_{}.json".format(cfg.dataset.name, split, cfg.dataset.test_split), "w") as f:
            json.dump(score_dict_by_name, f, indent=4)
    with open("metrics_{}_{}.json".format(cfg.dataset.name, split), "w") as f:
        json.dump(score_dict_by_name, f, indent=4)

if __name__ == "__main__":
    main()

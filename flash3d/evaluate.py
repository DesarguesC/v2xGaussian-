import os
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
from datasets.util import create_datasets
from misc.util import add_source_frame_id
from misc.visualise_3d import save_ply


def get_model_instance(model):
    """
    unwraps model from EMA object
    """
    return model.ema_model if type(model).__name__ == "EMA" else model

def operator(input_image: np.ndarray) -> dict:
    # input_image = cv2.imread(...) or Image.open(...)
    # TODO: how do I transfer the input_image to a correct type ?

    # see: models.encoder.unidepth_encoder.py -> class: UniDepthExtended.forward
    # see: models.model.py -> class: GaussianPredictor.compute_gauss_means ❎ 不用看，内部自己操作的

    """
        inputs[('unidepth', 0, 0)]
        inputs[("K_src", 0)]
        inputs["color_aug", 0, 0]

        TODO: 看在KITTI上的实现
        construction method: see "__getitem__" of the class in datasets/{re10k.py, dataset.py, kitti.py}
    """

    target_dict = {
        'target_frame_ids': ...,

    }



    return input_image


def evaluate(model, cfg, evaluator, input_image, device=None, save_vis=False):
    model_model = get_model_instance(model)
    model_model.set_eval()

    inputs = operator(input_image)

    with torch.no_grad():
        if device is not None:
            to_device(inputs, device)
        # inputs["target_frame_ids"] = target_frame_ids
        outputs = model(inputs) # dict

    for f_id in score_dict.keys():
        pred = outputs[('color_gauss', f_id, 0)]
        if cfg.dataset.name == "dtu":
            gt = inputs[('color_orig_res', f_id, 0)]
            pred = TF.resize(pred, gt.shape[-2:])
        else:
            gt = inputs[('color', f_id, 0)]
        # should work in for B>1, however be careful of reduction
        out = evaluator(pred, gt)
        if save_vis:
            save_ply(outputs, out_dir_ply / f"{f_id}.ply", gaussians_per_pixel=model.cfg.model.gaussians_per_pixel)
            pred = pred[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            gt = gt[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            plt.imsave(str(out_pred_dir / f"{f_id:03}.png"), pred)
            plt.imsave(str(out_gt_dir / f"{f_id:03}.png"), gt)
        for metric_name, v in out.items():
            score_dict[f_id][metric_name].append(v)



    metric_names = ["psnr", "ssim", "lpips"]
    score_dict_by_name = {}
    for f_id in score_dict.keys():
        score_dict_by_name[score_dict[f_id]["name"]] = {}
        for metric_name in metric_names:
            # compute mean
            score_dict[f_id][metric_name] = sum(score_dict[f_id][metric_name]) / len(score_dict[f_id][metric_name])
            # original dict has frame ids as integers, for json out dict we want to change them
            # to the meaningful names stored in dict
            score_dict_by_name[score_dict[f_id]["name"]][metric_name] = score_dict[f_id][metric_name]

    for metric in metric_names:
        vals = [score_dict_by_name[f_id][metric] for f_id in eval_frames]
        print(f"{metric}:", np.mean(np.array(vals)))

    return score_dict_by_name


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(opt, cfg: DictConfig):

    if not os.path.exists(opt.save_dir): os.mkdir(opt.save_dir)
    output_dir = os.path.join(opt.save_dir, 'flash3d') # temporary output of Flash3D

    ori_dir = os.getcwd() # save for standby
    os.chdir(output_dir)
    print(f"Saving dir: {output_dir} | Flash3D Working dir: {output_dir}")

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
    dataset, dataloader = inference(opt.input_img) #  create_datasets(cfg, split=split)
    score_dict_by_name = evaluate(model, cfg, evaluator, dataloader, 
                                  device=device, save_vis=save_vis)
    print(json.dumps(score_dict_by_name, indent=4))
    if cfg.dataset.name=="re10k":
        with open("metrics_{}_{}_{}.json".format(cfg.dataset.name, split, cfg.dataset.test_split), "w") as f:
            json.dump(score_dict_by_name, f, indent=4)
    with open("metrics_{}_{}.json".format(cfg.dataset.name, split), "w") as f:
        json.dump(score_dict_by_name, f, indent=4)
    

if __name__ == "__main__":
    main()

import torch, cv2, os, glob, subprocess, random, yaml, sys, pdb
import numpy as np
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from seem.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from .modeling.language.loss import vl_similarity
from .utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.trainers import load_checkpoint
from lama.saicinpainting.evaluation.data import pad_tensor_to_modulo

from seem.utils.arguments import load_opt_from_config_files
from seem.modeling.BaseModel import BaseModel
from seem.modeling import build_model
from seem.utils.constants import COCO_PANOPTIC_CLASSES


metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES] + ["others"]
colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]



def preload_seem_detector(opt, preloaded_seem_detector = None):
    if preloaded_seem_detector is None:
        cfg = load_opt_from_config_files([opt.seem_cfg])
        cfg['device'] = opt.device
        try:
            seem_model = BaseModel(cfg, build_model(cfg)).from_pretrained(opt.seem_ckpt).eval().cuda() # remember to compile SEEM
        except Exception as err:
            print('debug')
            print(f'[INFO]: {err}')
            pdb.set_trace()
    else:
        cfg = preloaded_seem_detector['cfg']
        seem_model = preloaded_seem_detector['seem_model']

    with torch.no_grad():
        seem_model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"],
                                                                                 is_eval=True)
    seem_model.model.task_switch['spatial'] = False
    seem_model.model.task_switch['visual'] = False
    seem_model.model.task_switch['grounding'] = False
    seem_model.model.task_switch['audio'] = False
    seem_model.model.task_switch['grounding'] = True

    preloaded_seem_detector = seem_model
    return preloaded_seem_detector, cfg

def preload_lama_remover(opt, preloaded_lama_dict = None):
    if preloaded_lama_dict is not None: return preloaded_lama_dict

    # seed_everything(opt.seed)
    predict_config = OmegaConf.load(opt.lama_config)
    predict_config.model.path = opt.lama_ckpt
    device = torch.device(opt.device)
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)

    return {
        'model': model,
        'config': predict_config
    }


def dilate_mask(mask, dilate_factor=15):
    """
    if np.max(mask) <= 1.:
        mask *= 255.
    """
    # `inpaint_img_with_lama` will finish this
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask


@torch.no_grad()
def inpaint_img_with_lama(
        img: np.ndarray,
        mask: np.ndarray,
        mod=8,
        device="cuda",
        preloaded_lama_remover=None
):
    if isinstance(mask, list):
        print(f'len(mask) = {len(mask)}')
        for i in range(len(mask)):
            print(f'mask[{i}].shape = {mask[i].shape}')
    else:
        print(f'mask.shape = {mask.shape}')

    # for mask_ in mask:
    # assert len(mask_.shape) == 2
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    if np.max(mask) == 1:
        mask = mask * 255
    device = torch.device(device)

    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()
    print(' ' * 6 + '-' * 9 + 'loading lama' + '-' * 9)

    model = preloaded_lama_remover['model']
    predict_config = preloaded_lama_remover['config']

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res




def process_seem_outputs(temperature, results, extra):

    pred_masks = results['pred_masks'][0]

    v_emb = results['pred_captions'][0]
    t_emb = extra['grounding_class']
    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

    # out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
    matched_id = vl_similarity(v_emb, t_emb, temperature=temperature).max(0)[1]
    pred_masks_pos = pred_masks[matched_id, :, :]
    # assert isinstance(pred_masks_pos, list) and len(pred_masks_pos) == 1, f'len(pred_masks_pos) = {pred_masks_pos}'
    return pred_masks, t_emb, v_emb, pred_masks_pos


def FG_remove(opt, img, reftxt = 'Cars', preloaded_seem_detector = None, preloaded_lama_dict = None, dilate_kernel_size = 15):
    # img: np.array -> [H W 3]
    seem_model, seem_cfg = preload_seem_detector(opt, preloaded_seem_detector)
    # sys.exit(-1)
    preloaded_lama_dict = preload_lama_remover(opt, preloaded_lama_dict)

    height, width, _ = img.shape
    img_ori = np.asarray(img).copy()
    img = torch.from_numpy(img_ori).permute(2, 0, 1).cuda()
    visual = Visualizer(img, metadata=metadata)

    data = {"image": img, "height": height, "width": width}
    data['text'] = reftxt # flexible targets
    batch_inputs = [data]

    results, image_size, extra = seem_model.model.evaluate_demo(batch_inputs)
    temperature = seem_model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    *_, pred_masks_pos = process_seem_outputs(temperature, results, extra)

    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0, :, :data['height'],
                      :data['width']] > 0.0).float().cpu().numpy() # np.ndarray -> [3 H w]
    # mask queried from text
    # pred_box_pos = None
    demo = visual.draw_binary_mask(pred_masks_pos.squeeze(), text=reftxt)  # rgb Image
    res = demo.get_image() # visualized with [id2name]

    target_mask_list = [dilate_mask(a_mask, dilate_kernel_size) for a_mask in pred_masks_pos]

    # remove forground <CAR>
    img_inpainted = inpaint_img_with_lama(
        img = img_ori, mask = target_mask_list[0], mod = 8, device = opt.device, preloaded_lama_remover = preloaded_lama_dict
    ) # -> np.array([H W 3]) | cv2.imwrite: cv2.cvtColor(np.uint8(img_inpainted), cv2.COLOR_RGB2BGR)

    seg_mask = np.concatenate([np.expand_dims(tar, axis=0) for tar in target_mask_list], axis=0)
    print(f'seg_mask = {seg_mask.shape}')


    return Image.fromarray(res), seg_mask










import os, glob, cv2, argparse, torch, rembg, sys
import numpy as np
from PIL import Image
from seem.masks import FG_remove, preload_seem_detector, preload_lama_remover

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="./data/test1.jpg", type=str, help="path to image (png, jpeg, etc.)")
    # parser.add_argument('--model', default='u2net', type=str, help="rembg model, see https://github.com/danielgatis/rembg#models")
    parser.add_argument('--size', default=256, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0.2, type=float, help="output border ratio")
    parser.add_argument('--recenter', type=bool, default=True, help="recenter, potentially not helpful for multiview zero123")
    # SEEM
    parser.add_argument('--seem_ckpt', type=str, default="../Tools/SEEM/seem_focall_v0.pt", help='restore where the SEEM & LaMa model locates')
    parser.add_argument('--seem_cfg', type=str, default="seem/configs/seem/focall_unicl_lang_demo.yaml")
    # LaMa
    parser.add_argument('--lama_ckpt', type=str, default='../Tools/LaMa/', help='actually path to lama ckpt base folder, ckpt specified in config files')
    parser.add_argument('--lama_cfg', type=str, default='./configs/lama_default.yaml', help='path to lama inpainting config path')

    #Outputs
    parser.add_argument('--results', type=str, default='../v2x-outputs/pre-process/', help='result direction')

    opt = parser.parse_args()
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

    if os.path.isdir(opt.path):
        print(f'[INFO] processing directory {opt.path}...')
        files = glob.glob(f'{opt.path}/*')
        # out_dir = opt.path
    else: # isfile
        files = [opt.path]
        # out_dir = os.path.dirname(opt.results)


    # preloaded_seem_detector = preload_seem_detector(opt)

    from seem.utils.arguments import load_opt_from_config_files
    from seem.modeling.BaseModel import BaseModel
    from seem.modeling import build_model

    cfg = load_opt_from_config_files([opt.seem_cfg])
    cfg['device'] = opt.device

    while True:
        try:
            built_model = build_model(cfg)
            seem_model = BaseModel(cfg, built_model)
            seem_model = seem_model.from_pretrained(opt.seem_ckpt).eval().cuda()  # remember to compile SEEM

            break
        except Exception as err:
            print(err)




    with torch.no_grad():
        seem_model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"],
                                                                                 is_eval=True)
    seem_model.model.task_switch['spatial'] = False
    seem_model.model.task_switch['visual'] = False
    seem_model.model.task_switch['grounding'] = False
    seem_model.model.task_switch['audio'] = False
    seem_model.model.task_switch['grounding'] = True

    preloaded_seem_detector = seem_model



    preloaded_lama_dict = preload_lama_remover(opt)


    print(f'files: {files}')
    for file in files:

        out_base = os.path.basename(file).split('.')[0]
        out_rgba = os.path.join(opt.results, out_base + '_rgba')
        cnt = 0
        while os.path.exists(f'{out_rgba}_{cnt}.png'):
            cnt += 1
        out_rgba = f'{out_rgba}_{cnt}.png'

        # load image
        print(f'[INFO] loading image {file}...')
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED) # read
        
        # TODO: use seem to remove foreground
        print(f'[INFO] background removal...')


        carved_image, mask = FG_remove(opt = opt, img = image, preloaded_seem_detector=preloaded_seem_detector, preloaded_lama_dict=preloaded_lama_dict)
        # exit(-1)
        # TODO: save intermediate results
        cv2.imwrite(os.path.join(opt.results, 'remove/r.jpg'), cv2.cvtColor(np.uint8(carved_image), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(opt.results, 'remove/m.jpg'), cv2.cvtColor(np.uint8(mask), cv2.COLOR_RGB2BGR))
        exit(-1)

    print('\nDone.')

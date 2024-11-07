import gradio as gr
from PIL import Image
import os

# path_list = [
#     '/NEW_EDS/JJ_Group/ChenD/v2x-outputs/pre-process/remove/mask.jpg',
#     '/NEW_EDS/JJ_Group/ChenD/v2x-outputs/pre-process/remove/res.jpg',
#     '/NEW_EDS/JJ_Group/ChenD/v2x-outputs/pre-process/remove/removed-bg.jpg',
#     '/NEW_EDS/JJ_Group/ChenD/v2x-outputs/pre-process/remove/removed-fg.jpg',
#     '/NEW_EDS/JJ_Group/ChenD/V2X-Gaussian/data/depth/projected_pcd-fg.jpg',
#     '/NEW_EDS/JJ_Group/ChenD/v2x-outputs/pre-process/colored_pred_depth-fg.jpg',
#     '/NEW_EDS/JJ_Group/ChenD/v2x-outputs/pre-process/colored_pred_init-fg.jpg',
#     '/NEW_EDS/JJ_Group/ChenD/V2X-Gaussian/data/depth/projected_pcd-bg.jpg',
#     '/NEW_EDS/JJ_Group/ChenD/v2x-outputs/pre-process/colored_pred_depth-bg.jpg',
#     '/NEW_EDS/JJ_Group/ChenD/v2x-outputs/pre-process/colored_pred_init-bg.jpg'
# ]

path_list = [
    './debug/pred_dep_0',
    './debug/pred_dep_0/depth_image',
    './debug/pred_dep_1',
    './debug/pred_dep_1/depth_image',
    './debug/side-pred-0/view.jpg',
    './debug/side-pred-1/view.jpg',

    '../dair-test/weights/flash3d/gt',
    '../dair-test/weights/flash3d/pred'
]


"""
debug
-------pred_dep_0
    |       |
    |       |__fg_mask.jpg, rgb.jpg
    |       |__depth_image
    |               |
    |               |__colored_init.jpg, colored_pred_all.jpg, pred.jpg
    |
    |__pred_dep_1
    |       |
    |       |__ ...
    |
    |__side-pred-0
    |       |__view.jpg
    |
    |__side-pred-1
            |__view.jpg

"""

def load_image():
    mask1 = Image.open(os.path.join(path_list[0], 'fg_mask.jpg'))
    mask2 = Image.open(os.path.join(path_list[2], 'fg_mask.jpg'))

    rgb1 = Image.open(os.path.join(path_list[0], 'rgb.jpg'))
    rgb2 = Image.open(os.path.join(path_list[2], 'rgb.jpg'))

    dep_1_1, dep_1_2, dep_1_3 = Image.open(os.path.join(path_list[1], 'colored_init.jpg')), \
        Image.open(os.path.join(path_list[1], 'colored_pred_all.jpg')), \
        Image.open(os.path.join(path_list[1], 'pred.jpg'))

    dep_2_1, dep_2_2, dep_2_3 = Image.open(os.path.join(path_list[3], 'colored_init.jpg')), \
        Image.open(os.path.join(path_list[3], 'colored_pred_all.jpg')), \
        Image.open(os.path.join(path_list[3], 'pred.jpg'))

    flash3d_gt_f_0, flash3d_gt_f_s0 = Image.open(os.path.join(path_list[6], '000.png')), \
        Image.open(os.path.join(path_list[6], 's00.png'))
    flash3d_pred_f_0, flash3d_pred_f_s0 = Image.open(os.path.join(path_list[7], '000.png')), \
        Image.open(os.path.join(path_list[7], 's00.png'))

    side_view_1, side_view_2 = Image.open(path_list[4]), Image.open(path_list[5])

    return (rgb1, mask1, dep_1_1, dep_1_2, dep_1_3, rgb2, mask2, \
                dep_2_1, dep_2_2, dep_2_3, side_view_1, side_view_2,\
            flash3d_gt_f_0, flash3d_gt_f_s0, flash3d_pred_f_0, flash3d_pred_f_s0)

def main():
    with gr.Blocks() as demo:
        button = gr.Button("Load")
        with gr.Row():
            rgb1 = gr.Image(label=os.path.join(path_list[0], 'rgb.jpg'))
            mask1 = gr.Image(label=os.path.join(path_list[0], 'fg_mask.jpg'))
        with gr.Row():
            dep_1_1 = gr.Image(label=os.path.join(path_list[1], 'colored_init.jpg'))
            dep_1_2 = gr.Image(label=os.path.join(path_list[1], 'colored_pred_all.jpg'))
            dep_1_3 = gr.Image(label=os.path.join(path_list[1], 'pred.jpg'))

        with gr.Row():
            rgb2 = gr.Image(label=os.path.join(path_list[2], 'rgb.jpg'))
            mask2 = gr.Image(label=os.path.join(path_list[2], 'fg_mask.jpg'))
        with gr.Row():
            dep_2_1 = gr.Image(label=os.path.join(path_list[3], 'colored_init.jpg'))
            dep_2_2 = gr.Image(label=os.path.join(path_list[3], 'colored_pred_all.jpg'))
            dep_2_3 = gr.Image(label=os.path.join(path_list[3], 'pred.jpg'))

        with gr.Row():
            side_view_1 = gr.Image(label=path_list[4])
            side_view_2 = gr.Image(label=path_list[5])

        with gr.Row():
            flash3d_gt_f_0, flash3d_pred_f_0 = gr.Image(label=os.path.join(path_list[6], '000.png')), \
                gr.Image(label=os.path.join(path_list[7], '000.png'))

        with gr.Row():
            flash3d_gt_f_s0, flash3d_pred_f_s0 = gr.Image(label=os.path.join(path_list[6], 's00.png')), \
                gr.Image(label=os.path.join(path_list[7], 's00.png'))

        button.click(fn=load_image,
                     inputs=[],
                     outputs=[
                         rgb1, mask1, dep_1_1, dep_1_2, dep_1_3,
                         rgb2, mask2, dep_2_1, dep_2_2, dep_2_3,
                         side_view_1, side_view_2,
                         flash3d_gt_f_0, flash3d_gt_f_s0, flash3d_pred_f_0, flash3d_pred_f_s0
                     ])

    # 启动 Gradio 界面
    demo.launch(server_name="127.0.0.1", server_port=6006)
    print('launched at \'http://127.0.0.1:6006\'')


if __name__ == '__main__':
    main()
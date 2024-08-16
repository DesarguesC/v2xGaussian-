import gradio as gr
from PIL import Image
import os

path_list = [
    '/NEW_EDS/JJ_Group/ChenD/v2x-outputs/pre-process/remove/mask.jpg',
    '/NEW_EDS/JJ_Group/ChenD/v2x-outputs/pre-process/remove/res.jpg',
    '/NEW_EDS/JJ_Group/ChenD/v2x-outputs/pre-process/remove/removed.jpg', 
    '/NEW_EDS/JJ_Group/ChenD/V2X-Gaussian/data/depth/projected_pcd.jpg',
    '/NEW_EDS/JJ_Group/ChenD/V2X-Gaussian/data/depth/colored_pred_depth.jpg',
    '/NEW_EDS/JJ_Group/ChenD/V2X-Gaussian/data/depth/colored_pred_init.jpg'
]

# 定义函数以读取并返回图片
def load_images():
    # 读取当前文件夹下的图片

    # img_list = [Image.open(file) for file in path_list]
    # for file in path_list:
    # img1 = img1.resize((img1.size[0]//2, img1.size[1]//2), Image.Resampling.BILINEAR)
    # img2 = img2.resize((img2.size[0]//2, img2.size[1]//2), Image.Resampling.BILINEAR)
    # img3 = img3.resize((img3.size[0]//2, img3.size[1]//2), Image.Resampling.BILINEAR)

    return Image.open(path_list[0]), Image.open(path_list[1]), Image.open(path_list[2]), Image.open(path_list[3]), Image.open(path_list[4]), Image.open(path_list[5])

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Button("Load Images").click(load_images, outputs=[gr.Image(type='pil') for i in path_list])

# 启动 Gradio 界面
demo.launch(server_name="127.0.0.1", server_port=6006)
print('launched at \'http://127.0.0.1:6006\'')


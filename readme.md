## SEEM
```bash
sudo apt-get update
sudo apt-get install ffmpeg
cd seem/modeling/vision/encoder/ops && python setup.py build install && cd ../../../../../
```

[//]: # (## LaMa)

[//]: # (```bash)

[//]: # ()
[//]: # (```)

## CompletionFormer
```bash
git clone https://github.com/NVIDIA/apex
cd apex
git reset --hard 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 

cd lidar2dep/model/deformconv && python setup.py build install && cd ../../../
```


## mannually install
Environment
```bash
conda create -n v2x python=3.9.17
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Packages from GitHub
```bash
pip install git+https://github.com/NVlabs/nvdiffrast/ && \
pip install git+https://github.com/ashawkey/kiuikit && \
pip install git+https://github.com/bytedance/MVDream
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream && \
pip install git+https://github.com/MaureenZOU/detectron2-xyz.git && \
pip install -e git+https://github.com/DesarguesC/kornia@master#egg=kornia
```



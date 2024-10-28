## Environment
manually installation
```bash
conda create -n v2x python=3.10.15
conda activate v2x
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Packages from GitHub
```bash
pip install git+https://github.com/NVlabs/nvdiffrast/ && \
pip install git+https://github.com/ashawkey/kiuikit.git && \
pip install git+https://github.com/bytedance/MVDream && \
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream && \
pip install git+https://github.com/MaureenZOU/detectron2-xyz.git && \
pip install git+https://github.com/DesarguesC/kornia@master#egg=kornia && \
pip install git+https://github.com/facebookresearch/pytorch3d.git
```


## SEEM
```bash
sudo apt-get update
sudo apt-get install ffmpeg
cd seem/modeling/vision/encoder/ops && python setup.py build install

# if there is a need to re-build
rm -rf build
cd ../../../../../
```

## CompletionFormer
```bash
git clone https://github.com/NVIDIA/apex
cd apex
git reset --hard 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a

# pip > 23.1
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 

# build DCN module
cd lidar2dep/model/deformconv && python setup.py build install && cd ../../../
#mmcv
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -v -e .
# mmseg
cd ../ && git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

## DRGS
```bash
pip install -e submodules/diff-gaussian-rasterization-depth-acc
```


Note that when install "apex", there is probably encountered with an error "cannot import name 'container_abcs' from 'torch._six'", 
which can be solved by directly amend the file where import error occurred.



## LLM

put your OpenAI/Anthropic Token in 'key.csv', Anthropic Claude is highly recommended.


## Debug-Mode

For debugging, run the command
```bash
export READ_ONLY=1 && export CUDA_LAUNCH_BLOCKING=1 && srun python drgs.py --downsample <downsample scale>
```



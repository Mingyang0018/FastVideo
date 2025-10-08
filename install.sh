#!/bin/bash
# install.sh

set -e

# 创建并激活 conda 环境
conda env create -f environment.yaml || conda env update -f environment.yaml
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fast_video

# 使用 PyTorch 官方 wheels 源安装 torch + torchvision（示例为 CUDA 12.9）
python -m pip install --upgrade --trusted-host download.pytorch.org \
  --index-url https://download.pytorch.org/whl/cu129 \
  torch torchvision
  
# 安装 pip 依赖
pip install -r requirements.txt || true

# 克隆 MultiTalk
if [ ! -d "MultiTalk" ]; then
    git clone https://github.com/MeiGen-AI/MultiTalk.git
fi

# 下载模型
huggingface-cli download black-forest-labs/FLUX.1-dev ./FLUX.1-dev --resume-download --extract || true
huggingface-cli download black-forest-labs/FLUX.1-Krea-dev ./FLUX.1-Krea-dev --resume-download --extract || true
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers ./Wan2.2-TI2V-5B-Diffusers --resume-download --extract || true

# 安装 FFmpeg (如未安装)
conda install -c conda-forge ffmpeg -y

# 安装 librosa (如未安装)
conda install -c conda-forge librosa -y

# 下载https://github.com/MeiGen-AI/MultiTalk.git
git clone https://github.com/MeiGen-AI/MultiTalk.git

# 复制 environment01.yaml, fastapi_multitalk.py 到 MultiTalk 目录
cp environment01.yaml, fastapi_multitalk.py ./MultiTalk/
# 创建并激活 conda 环境
conda env create -f ./MultiTalk/environment01.yaml || conda env update -f ./MultiTalk/environment01.yaml
source $(conda info --base)/etc/profile.d/conda.sh
conda activate multitalk

# 使用 PyTorch 官方 wheels 源安装 torch + torchvision
python -m pip install --upgrade --trusted-host download.pytorch.org \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

python -m pip install --upgrade --trusted-host download.pytorch.org \
  --index-url https://download.pytorch.org/whl/cu121 \
  -U xformers==0.0.28 || true
python -m pip install flash_attn==2.7.4.post1 || true

conda install -c conda-forge libstdcxx-ng=12.2.0 -y
conda install -c conda-forge librosa -y
conda install -c conda-forge ffmpeg -y

echo "✅ 环境和模型安装完成，工作流示例为 my_pipeline.ipynb，web程序为 streamlit_app.py"

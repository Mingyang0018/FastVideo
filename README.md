# FastVideo 自动化生成短剧视频

本项目支持自动化生成短剧视频，实现从主题到多角色交互的短剧视频的全流程自动化，包含生成剧本、合成语音、生成角色、生成视频、合成短剧。

## 环境安装

1. 安装 Miniconda 或 Anaconda
2. 运行 install.sh 一键安装环境和依赖

```bash
bash install.sh
```

## 主要依赖

- Python 3.12
- PyTorch (CUDA 12)
- diffusers, transformers, moviepy, ultralytics, gradio, edge-tts, rembg, opencv-python, etc.
- FFmpeg, librosa

## 模型下载

install.sh 会自动下载以下模型：

- black-forest-labs/FLUX.1-dev
- black-forest-labs/FLUX.1-Krea-dev
- Wan-AI/Wan2.2-TI2V-5B-Diffusers
- MultiTalk (git clone)

## 启动web程序

```bash
streamlit run streamlit_app.py
```

---

# 目录结构

```
./fast_video/
├── README.md              # 项目说明文件
├── streamlit_app.py       # Streamlit Web 应用
├── main.py                # 主程序
├── fastapi_text2image.py  # FastAPI 图像生成接口
├── fastapi_image2image.py # FastAPI 图像到图像生成接口
├── fastapi_image2video.py # FastAPI 图像到视频生成接口
├── yolov8n-seg.pt         # YOLO预训练模型
├── install.sh             # 安装脚本
├── environment.yaml       # Conda 环境配置
├── requirements.txt       # Python 依赖
├── FLUX.1-dev/            # FLUX 模型文件
├── FLUX.1-Krea-dev/       # Krea 模型文件
├── Wan2.2-TI2V-5B-Diffusers/ # Wan 模型文件
├── MultiTalk/             # MultiTalk 模型文件
└── output/                # 生成的文件和视频存放目录

```

---

# 参考

- MultiTalk: https://github.com/MeiGen-AI/MultiTalk
- Huggingface: https://huggingface.co

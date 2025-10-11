# FastVideo 自动化生成短剧视频

本项目支持自动化生成短剧视频，实现从主题到多角色交互的短剧视频的全流程自动化，包含生成剧本、合成语音、生成角色、生成视频、合成短剧。

![img](https://github.com/Mingyang0018/FastVideo/releases/download/v1.0/Screenshot.png)

## 算法框架

```mermaid
graph LR;
    A[主题输入] --> B[DeepSeek / ChatGPT 生成初版剧本];
    B --> C[迭代优化剧本];
    C --> C1[生成语音];
    C1 --> D[text2image 生成角色与背景];
    D --> E[image2image 生成分镜角色与背景];
    E --> F[image2video 生成视频片段];
    F --> G[拼接视频];
    G --> H[合成短剧视频];

    %% 辅助分支说明
    subgraph DeepSeek / ChatGPT
        B
        C
	C1
    end

    subgraph Diffusion
        D
        E
        F
    end

```

## 短剧示例

<div align="center">
    <video width="40%" controls>
        <source src="https://github.com/Mingyang0018/FastVideo/releases/download/v1.0/FastVideo_20250924171430.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <video width="40%" controls>
        <source src="https://github.com/Mingyang0018/FastVideo/releases/download/v1.0/FastVideo_20251006202718.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

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
- Huggingface: https://huggingface.co `<td>`

`<video src="https://github.com/Mingyang0018/FastVideo/releases/download/v1.0/FastVideo_20250924171430.mp4" width="450" controls loop>`

`</video>`

`</td>`
      `<td>`
          `<video src="https://github.com/Mingyang0018/FastVideo/releases/download/v1.0/FastVideo_20251006202718.mp4" width="450" controls loop></video>`
      `</td>`

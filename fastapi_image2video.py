from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import DiffusionPipeline, WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import load_image, export_to_video
import torch
import gc
from typing import Optional
from PIL import Image
import io
import base64
import os
import sys
import signal

app = FastAPI(title="image2video API")

# ------------------- 输入数据结构 -------------------
class T2IRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    image_base64: str
    width: int = 704
    height: int = 1280
    num_frames: int = 81
    guidance_scale: float =5.0
    num_inference_steps: int = 30
    output_path: str = 'video.mp4'

class ModelRequest(BaseModel):
    model_id: str

pipe: WanImageToVideoPipeline = None

@app.post("/load_model")
def load_model(req: ModelRequest):
    global pipe
    if pipe is not None:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        print("♻️ 旧模型已卸载")

    print(f"🚀 正在加载模型: {req.model_id}")
    vae = AutoencoderKLWan.from_pretrained(
        req.model_id, subfolder="vae", torch_dtype=torch.float16
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        req.model_id,
        vae=vae,
        torch_dtype=torch.float16,
        # device_map="balanced"
    )
    pipe.enable_model_cpu_offload()
    # pipe = DiffusionPipeline.from_pretrained(
    #     req.model_id, 
    #     # torch_dtype=torch.float16,
    #     device_map="balanced"
    # )
    pipe.enable_attention_slicing()
    print(f"✅ 模型 {req.model_id} 加载完成")
    return {"status": "ok", "model_id": req.model_id}

# 校正分辨率为64的倍数
def round_to_multiple(x, base=64):
    return (x // base) * base

@app.post("/image2video")
def image2video(req: T2IRequest):
    global pipe
    if pipe is None:
        return {"error": "模型未加载"}
    
    app.state.busy = True
    try:
        image_bytes = base64.b64decode(req.image_base64)
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # 自动修正 width/height，避免 tensor mismatch
        width = round_to_multiple(req.width, 64)
        height = round_to_multiple(req.height, 64)

        output = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=init_image,
            width=width,
            height=height,
            num_frames=req.num_frames,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps
        ).frames[0]
        output_dir = os.path.dirname(req.output_path)
        os.makedirs(output_dir, exist_ok=True)
        export_to_video(output, req.output_path, fps=24) 

        # 清理 GPU
        torch.cuda.empty_cache()
        gc.collect()
        print(f"✅ 视频已保存到 {req.output_path}")
        return {"status": "ok", "video_path": req.output_path}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # 释放状态 & 清理资源
        app.state.busy = False
        for var in ["image", "init_image", "buffered", "image_bytes", "output"]:
            if var in locals():
                del locals()[var]
        gc.collect()
        torch.cuda.empty_cache()

@app.post("/shutdown")
def shutdown_server():
    """
    远程关闭 uvicorn 服务 + 释放显存
    """
    global pipe
    if pipe is not None:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ 模型已释放，显存回收完成")

    print("🛑 收到关闭请求，正在退出 FastAPI...")
    # 彻底退出 uvicorn
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting down"}

@app.get("/busy")
def busy_status():
    return {"busy": getattr(app.state, "busy", False)}

@app.get("/health")
def health_check():
    return {"status": "ok"}
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
import gc
from typing import Optional
from PIL import Image
import io
import base64
import os
import sys
import signal

app = FastAPI(title="Text2Image API")

# ------------------- 输入数据结构 -------------------
class T2IRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 25

class ModelRequest(BaseModel):
    model_id: str

pipe: DiffusionPipeline = None

@app.post("/load_model")
def load_model(req: ModelRequest):
    global pipe
    if pipe is not None:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        print("♻️ 旧模型已卸载")

    print(f"🚀 正在加载模型: {req.model_id}")
    try:
        # 根据系统 GPU 数量自动选择设备策略：
        # - 0 GPU: 使用 CPU
        # - 1 GPU: 将整个 pipeline 放到单个 GPU（精度使用 float16 以节省显存）
        # - 多 GPU: 使用 device_map="balanced" 让 accelerate/transformers 在多卡上分配
        cuda_count = torch.cuda.device_count()
        if cuda_count == 0:
            print("未检测到 GPU，使用 CPU 加载模型")
            pipe = DiffusionPipeline.from_pretrained(
                req.model_id,
                local_files_only=True,
            )
            pipe.to("cpu")
        elif cuda_count == 1:
            print("检测到 1 个 GPU，使用单卡（cuda:0）")
            pipe = DiffusionPipeline.from_pretrained(
                req.model_id,
                # torch_dtype=torch.float16,
                device_map="balanced",
                local_files_only=True,
            )
            # pipe.to("cuda")
        else:
            print(f"检测到 {cuda_count} 个 GPU，使用 device_map='balanced' 分配到多卡")
            pipe = DiffusionPipeline.from_pretrained(
                req.model_id,
                device_map="balanced",
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            # pipe.parallelize()

        pipe.enable_attention_slicing()
        print(f"✅ 模型 {req.model_id} 加载完成")
        return {"status": "ok", "model_id": req.model_id}
    except Exception as e:
        import traceback
        print("❌ 模型加载失败:", e)
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/text2image")
def text2image(req: T2IRequest):
    global pipe
    if pipe is None:
        return {"error": "模型未加载"}
    
    app.state.busy = True
    try:
        image = pipe(
            req.prompt,
            negative_prompt=req.negative_prompt,
            num_images_per_prompt=1,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps
        ).images[0]
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"image_base64": img_str}
    except Exception as e:
        # 捕获异常并返回
        return {"error": str(e)}

    finally:
        # 释放状态 & 清理资源
        app.state.busy = False
        for var in ["image", "init_image", "buffered", "image_bytes"]:
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

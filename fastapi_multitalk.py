import subprocess
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import json
import librosa  # 用于读取音频时长
import soundfile as sf
import numpy as np
import gc
import torch
import signal

app = FastAPI(title="MultiTalk API", description="FastAPI wrapper for MultiTalk video generation")

class MultiTalkRequest(BaseModel):
    ckpt_dir: str = "weights/Wan2.1-I2V-14B-480P"
    wav2vec_dir: str = "weights/chinese-wav2vec2-base"
    input_json: str = "examples/single_example_1.json"
    lora_dir: Optional[str] = "weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
    lora_scale: Optional[float] = 1.2
    sample_text_guide_scale: float = 1.0
    sample_audio_guide_scale: float = 2.0
    sample_steps: int = 8
    mode: str = "streaming"
    num_persistent_param_in_dit: int = 0
    save_file: str = "single_long_fusionx_exp"
    sample_shift: int = 2
    # quant_dir: str = "weights/MeiGen-MultiTalk"
    
def pad_audio(audio_path: str, min_duration: float = 82/25) -> str:
    """补齐音频到 min_duration 秒，返回新音频路径"""
    try:
        # 读取音频
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        print('duration: ', duration)
        if duration >= min_duration:
            return audio_path  # 不需要补齐
        # 在尾部追加渐弱尾音
        fade_len = int(0.01 * sr)  # 10ms
        if fade_len < len(y):
            fade_out = np.linspace(1.0, 0.0, fade_len)
            y = np.concatenate([y, y[-fade_len:] * fade_out])

        # 补齐剩余静音
        duration_after_fade = len(y) / sr
        pad_len = int(max(0, (min_duration - duration_after_fade) * sr))
        if pad_len > 0:
            y = np.concatenate([y, np.zeros((pad_len,))])

        # 覆盖原始文件
        sf.write(audio_path, y, sr)
        return audio_path

    except Exception as e:
        print(f"⚠️ 音频补齐失败: {e}")
        return audio_path


@app.post("/generate")
def generate_video(req: MultiTalkRequest, stream_output=True):
    """
    Run MultiTalk pipeline to generate lip-sync video.
    """
    # 读取并修改 input_json，处理短音频
    with open(req.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cond_audio = data.get("cond_audio", {})
    for k, v in cond_audio.items():
        if os.path.exists(v):
            pad_audio(v, min_duration=82/25)  # 自动补齐短音频

    temp_audio_dir = tempfile.mkdtemp()  # 系统临时目录

    cmd = [
        "python", "generate_multitalk.py",
        "--ckpt_dir", req.ckpt_dir,
        "--wav2vec_dir", req.wav2vec_dir,
        "--input_json", req.input_json,
        "--sample_text_guide_scale", str(req.sample_text_guide_scale),
        "--sample_audio_guide_scale", str(req.sample_audio_guide_scale),
        "--sample_steps", str(req.sample_steps),
        "--mode", req.mode,
        "--num_persistent_param_in_dit", str(req.num_persistent_param_in_dit),
        "--use_teacache", 
        "--teacache_thresh", "0.5", 
        "--save_file", req.save_file,
        "--sample_shift", str(req.sample_shift),
        "--audio_save_dir", temp_audio_dir,
        # "--quant", "int8",
        # "--quant_dir", req.quant_dir
    ]

    # only add LoRA if provided
    if req.lora_dir:
        cmd += ["--lora_dir", req.lora_dir, "--lora_scale", str(req.lora_scale)]

    # if req.quant_dir:
    #     cmd += ["--quant", "int8", "--quant_dir", req.quant_dir]
    # cuda_count = torch.cuda.device_count()

    # 生成器函数，逐行读取子进程输出
    def run_and_stream():
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        line_count = 0

        yield f"\n cmd: {cmd}"
        for line in iter(process.stdout.readline, ''):
            if line_count % 10 == 0:
                yield line  # 🚀 逐行返回给客户端
            line_count += 1
        process.wait()
        yield f"\n[结束] returncode={process.returncode}\n"
        video_path = f"{req.save_file}.mp4"
        if os.path.exists(video_path):
            yield f"[成功] 视频已生成: {video_path}\n"
        else:
            yield "[失败] 视频未生成\n"
    if stream_output:
        return StreamingResponse(run_and_stream(), media_type="text/plain")
    else:
        try:
            os.makedirs(os.path.dirname(req.save_file), exist_ok=True)
            print("cmd: ", cmd)
            # result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # 实时打印子进程日志
            for line in iter(process.stdout.readline, ''):
                print(line, end="", flush=True)

            process.wait()
            if process.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Video generation failed, code={process.returncode}")

            video_path = f"{req.save_file}.mp4"
            if not os.path.exists(video_path):
                raise HTTPException(status_code=500, detail="Video generation failed. No output file found.")

            return {
                    "cmd": cmd,
                    "message": "Video generated successfully",
                    "video_file": video_path,
                }

        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500, 
                detail={
                "error": "MultiTalk failed",
                "stdout": e.stdout,
                "stderr": e.stderr,
                "cmd": e.cmd
            }
            )

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/shutdown")
def shutdown_server():
    """
    远程关闭 uvicorn 服务 + 释放显存
    """
    gc.collect()
    torch.cuda.empty_cache()
    print("🛑 收到关闭请求，正在退出 FastAPI...")
    # 彻底退出 uvicorn
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting down"}

@app.get("/busy")
def busy_status():
    return {"busy": getattr(app.state, "busy", False)}

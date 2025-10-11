# from unittest import result
from openai import OpenAI
import numpy as np
import edge_tts
from edge_tts import Communicate
from diffusers import DiffusionPipeline, WanImageToVideoPipeline, AutoencoderKLWan
# from IPython.display import display
import json
import os
from deep_translator import GoogleTranslator
import torch
from dotenv import load_dotenv
import gc
from rembg import remove
import io
import ast
import re
import cv2
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from ultralytics import YOLO
from diffusers.utils import load_image, export_to_video
import requests
import subprocess
import time
import base64
from datetime import datetime
import textwrap
import clip
import shutil
import tempfile
from dataclasses import dataclass, field, fields, asdict
from typing import List, Dict, Any
import asyncio
import sys
# import streamlit as st
from io import StringIO

class StreamlitLogger:
    def __init__(self, placeholder):
        self.placeholder = placeholder  # 前端展示容器
        self.log_buffer = StringIO()
        self.stdout = sys.stdout
        self.stderr = sys.stderr    

    def write(self, message):
        if message.strip():  # 忽略空行
            self.log_buffer.write(message + "\n")
            # 更新到前端
            self.placeholder.text(self.log_buffer.getvalue())

    def flush(self):
        pass  # 为了兼容 file-like object

# ----------------------- dataclass 封装 -----------------------
@dataclass
class BaseInfo:
    api_key: str = "xx"
    # HF_TOKEN: str = "xx"
    model_id: str = "deepseek-chat"
    keywords: str = "喜剧"
    # 基本信息
    theme: str = "默认主题"
    time_limit: int = 300
    language: str = "中文"
    culture_background: str = "Ancient Chinese"
    drama_style: str = "Animation effect"
    width: int = 720
    height: int = 1280
    text2image_model: str = "./FLUX.1-dev"
    image2image_model: str = "./FLUX.1-Kontext-dev"
    image2video_model: str = "./Wan2.2-TI2V-5B-Diffusers"
    extra: Dict[str, Any] = field(default_factory=dict)

    # 自动生成的属性
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    BASE_DIR: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
    OUTPUT_DIR: str = field(init=False)
    path_script: str = field(init=False)
    path_drama_data: str = field(init=False)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # 可选代理
    use_proxy: bool = False
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 7897

    def __post_init__(self):
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, f"output")
        # self.OUTPUT_DIR = os.path.join(self.BASE_DIR, f"output_{self.timestamp}")
        self.path_script = os.path.join(self.OUTPUT_DIR, "script.md")
        self.path_drama_data = os.path.join(self.OUTPUT_DIR, "drama_data.json")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.backends.cudnn.benchmark = True
        # if self.HF_TOKEN:
        #     os.environ["HF_TOKEN"] = self.HF_TOKEN
        # 可选：检查 api_key / HF_TOKEN 是否有效
        if not self.api_key:
            print("警告: api_key 为空, 请在实例化 BaseInfo 时传入。")
        # 代理
        if self.use_proxy:
            proxy_url = f"http://{self.proxy_host}:{self.proxy_port}"
            os.environ["http_proxy"] = proxy_url
            os.environ["https_proxy"] = proxy_url

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key)

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra[key] = value

@dataclass
class DramaData:
    base_info: BaseInfo = field(default_factory=BaseInfo)
    scripts: Dict[str, Any] = field(default_factory=dict)
    roles: Dict[str, Any] = field(default_factory=dict)

    # 保存/加载
    def save_json(self):
        with open(self.base_info.path_drama_data, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

    def load_json(self):
        if not os.path.exists(self.base_info.path_drama_data): 
            return None
        with open(self.base_info.path_drama_data, "r", encoding="utf-8") as f:
            data = json.load(f)
        base_info_data = data.get("base_info", {})

        # 只保留 BaseInfo 构造函数能接收的字段
        baseinfo_fields = {f.name for f in fields(BaseInfo) if f.init}
        base_info_data = {k: v for k, v in base_info_data.items() if k in baseinfo_fields}

        base_info = BaseInfo(**base_info_data)

        return self.__class__(
            base_info=base_info,
            scripts=data.get("scripts", {}),
            roles=data.get("roles", {})
        )
    # @classmethod
    # def load_json(cls, path: str):
    #     with open(path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     base_info_data = data.get("base_info", {})
    #     base_info = BaseInfo(**base_info_data)
    #     return cls(
    #         base_info=base_info,
    #         scripts=data.get("scripts", {}),
    #         roles=data.get("roles", {})
    #     )
    @staticmethod
    def translate_text(text):
        return GoogleTranslator(source='auto', target='en').translate(text)

class DramaPipeline:
    def __init__(self, data: DramaData, api_url: str = "http://127.0.0.1:5101"):
        self.data = data
        self.api_url = api_url
        self.client = OpenAI(api_key=self.data.base_info.api_key, base_url="https://api.deepseek.com")
        self.model_id=self.data.base_info.model_id
        self.output_video_path=None

    @staticmethod
    def safe_json_parse(text):
        """
        尝试把返回的字符串解析为 dict, 支持：
        1. 单引号的 Python dict 格式
        2. 标准 JSON 格式（带或不带 ```json 包裹）
        3. 自动去掉前后多余字符
        """

        if not text:
            return None

        # 去掉 ```json ``` 或 ``` 包裹
        text = re.sub(r"^```(json)?", "", text.strip())
        text = re.sub(r"```$", "", text.strip())

        # 先尝试 JSON 解析
        try:
            return json.loads(text)
        except Exception:
            pass

        # 如果是单引号的 dict, 尝试用 ast.literal_eval 转换
        try:
            return ast.literal_eval(text)
        except Exception:
            pass

        # 最后尝试：替换单引号为双引号, 再次 json.loads
        try:
            fixed = text.replace("'", '"')
            return json.loads(fixed)
        except Exception:
            pass

        # 全部失败
        return None

    def generate_response(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=False
        )

        try:
            content = response.choices[0].message.content
            result = self.safe_json_parse(content)
            if result is None:
                print("不是JSON格式")
                print(content)
                return content
            return result

        except Exception as e:
            print("❌ 无法获取API响应。", e)
            return {"正面提示词": "默认正面提示词", "负面提示词": "默认负面提示词"}

    # 1. 生成剧本
    def evaluate_script_with_llm(self, script_text: str) -> float:
        """
        使用 LLM 对剧本进行评分, 返回 0~1 的综合评分
        可自定义评分维度和权重
        """
        prompt = f"""
    请为以下短剧剧本评分, 满分1.0。
    评分维度：
    1. 逻辑性（剧情合理）
    2. 搞笑幽默性（充满笑点、对白幽默、动作夸张、悬念/反转）
    3. 完整性（剧情丰富+多场景+对话+动作+风景转场, 避免单一）
    4. 风格契合度（与指定风格贴合）
    5. 创新性（新颖, 不落俗套）
    6. 人物多样性（角色数量足够, 性格鲜明, 不同人物有独特对白）

    剧本：
    {script_text}

    输出 JSON, 例如：
    {{"逻辑性":0.62,"搞笑幽默性":0.7,"完整性":0.7,"风格契合度":0.85,"创新性":0.75}}
    只返回JSON结构本身, 不需要额外解释。
    """
        scores = self.generate_response([{"role":"user","content":prompt}])
        print(scores)
        try:
            # 综合评分权重, 可自定义
            weights = {
                "逻辑性": 2,
                "搞笑幽默性": 2,
                "完整性": 1,
                "风格契合度": 1,
                "创新性": 1,
                "人物多样性": 1
            }
            total_score = sum(scores.get(w,0)*weights.get(w,0) for w in weights)/sum(weights.values())
            print(total_score)
            return total_score
        except:
            # 出现解析异常时, 返回默认较低分
            return 0.5

    def generate_candidates(self, n=2):
        """
        生成 n 个候选剧本
        """
        candidates = []
        for i in range(n):
            messages=[
                {"role": "system", "content": "你是一个编剧, 擅长编写幽默、夸张、吸人眼球、剧情丰富的爆火的短剧。"},
                {"role": "user", 
                    "content": 
                    f'''
    请根据以下主题：{self.data.base_info.theme}, 短剧风格为：{self.data.base_info.drama_style}, 文化背景为：{self.data.base_info.culture_background}, 短剧关键词为：{self.data.base_info.keywords}，编写一个约{self.data.base_info.time_limit}分钟的完整的短剧剧本。
    并按场景分镜输出, 每个场景应包含角色对话、角色动作和风景转场描述。确保剧本幽默风趣、吸人眼球、剧情丰富, 用于生成爆火的短剧视频。
    要求：
    1. 按场景分镜, 每个场景有对话、动作和风景转场。
    2. 人物角色性格鲜明、吸人眼球。
    3. 对话幽默、机智, 有笑点或反转。
    4. 动作要夸张, 画面感强。
    5. 加入一些出人意料的情节。
    6. 风景转场自然过渡, 增加视觉冲击。
                    '''},
            ]
            script = self.generate_response(messages)
            candidates.append(script)
        return candidates

    def refine_candidates_with_llm(self, top_scripts):
        """
        对 top-K 剧本生成改写变体, 每个生成 2 个
        """
        refined = []
        for script in top_scripts:
            prompt = f"""
    请根据以下原剧本提出改进建议并生成新的剧本：
    - 增加搞笑幽默性和吸引力
    - 增加剧情和人物多样性
    - 修正逻辑漏洞
    - 增加场景完整性
    - 保持风格一致
    原剧本：
    {script}

    请生成新的完整剧本, 保持每个场景有对话、动作和风景转场描述。
    """
            for _ in range(1):
                new_script = self.generate_response([{"role":"user","content":prompt}])
                refined.append(new_script)
        return refined

    def generate_script_iterative(self):
        """
        完整的迭代生成+评分+改写流程
        """
        MAX_ITER = 1       # 最大迭代轮数
        INIT_CANDIDATES = 5  # 初始候选数量
        TOP_K = 2          # 每轮选取 top K
        STOP_SCORE = 0.9  # 停止阈值
        # ---- 1. 初代生成 ----
        print("初代生成候选剧本...\n")
        candidates = self.generate_candidates(n=INIT_CANDIDATES)
        top_scored = []
        for iter_idx in range(MAX_ITER):
            print(f"第 {iter_idx+1} 轮迭代评分...\n")
            # ---- 2. 评分 ----
            scored = [(self.evaluate_script_with_llm(script), script) for script in candidates]
            scored.sort(reverse=True, key=lambda x: x[0])
            top_scored.extend(scored[:TOP_K])
            print(f"本轮 top1 分数: {scored[0][0]:.3f}")
            if scored[0][0] >= STOP_SCORE:
                print(f"达到停止阈值 {STOP_SCORE}, 停止迭代。")
                break

            # ---- 3. 改写/变异 ----
            print("生成改写...\n")
            top_scripts = [s for score, s in scored[:TOP_K]]
            candidates = top_scripts + self.refine_candidates_with_llm(top_scripts)

        # ---- 4. 循环结束后, 再对最后一批 candidates 评分 ----
        print("最后一轮候选评分...\n")
        scored = [(self.evaluate_script_with_llm(script), script) for script in candidates]
        scored.sort(reverse=True, key=lambda x: x[0])
        top_scored.extend(scored[:TOP_K])
        top_scored.sort(reverse=True, key=lambda x: x[0])
        print(top_scored)
        best_score = top_scored[0][0]
        best_script = top_scored[0][1]
        # ---- 4. 保存 ----
        with open(self.data.base_info.path_script, "w", encoding="utf-8") as f:
            f.write(best_script)
        print(f"✅ 最终剧本生成完成, 分数{best_score}！")
        return best_script 
        
    def generate_roles_info(self):
        """
        将Markdown剧本通过LLM转换为结构化JSON剧本（两轮解析, 确保角色不遗漏）
        """
        path_script = self.data.base_info.path_script
        with open(path_script, "r", encoding="utf-8") as f:
            script_md = f.read()

        # ---------------- 第一次调用 ----------------
        messages1 = [
            {"role": "system", "content": "你是一个剧本解析专家, 擅长将剧本内容解析为结构化的JSON格式。"},
            {"role": "user", "content": f"""
    请将以下剧本中的所有的角色信息解析为JSON格式：

    {script_md}

    要求输出结构如下（注意保留中文字段）：

    {{
        '角色1': {{'describe': '描述内容'}}, 
        '角色2': {{'describe': '描述内容'}}
    }}

    注意事项：
    - describe用于描述角色的特征, 比如性格、外貌、穿着, 不超过50个字；
    - 保证获取所有人物角色的信息, 不要遗漏角色；
    只返回JSON结构本身, 不需要额外解释。
    """}
        ]
        roles_info_1 = self.generate_response(messages1)

        # ---------------- 第二次调用 ----------------
        messages2 = [
            {"role": "system", "content": "你是一个剧本解析专家, 擅长补充遗漏的角色信息。"},
            {"role": "user", "content": f"""
    以下是我已有的角色信息：
    {json.dumps(roles_info_1, ensure_ascii=False, indent=2)}

    请检查以下剧本, 补充遗漏的角色（如果没有遗漏则返回空对象 {{}}）：

    {script_md}

    输出格式与之前一致：
    {{
        '角色X': {{'describe': '描述内容'}}
    }}
    """}
        ]
        roles_info_2 = self.generate_response(messages2)

        # ---------------- 合并两次结果 ----------------
        merged_roles = {}
        for roles_dict in [roles_info_1, roles_info_2]:
            if roles_dict:
                for role, info in roles_dict.items():
                    # 如果已有该角色, 则保留第一次的描述
                    if role not in merged_roles:
                        merged_roles[role] = info

        return merged_roles

    def generate_json(self):
        """
        将Markdown剧本通过LLM转换为结构化JSON剧本
        """
        path_script = self.data.base_info.path_script
        roles_info = self.data.roles
        with open(path_script, "r", encoding="utf-8") as f:
            script_md = f.read()
        roles_json = json.dumps(roles_info, ensure_ascii=False)
        messages=[
            {"role": "system", "content": "你是一个剧本解析专家, 擅长将剧本内容解析为结构化的JSON格式。"},
            {"role": "user", "content": 
                f"""
    请将以下剧本内容解析为JSON格式：

    {script_md}

    要求输出结构如下（注意保留中文字段, 并包含 type 字段）：

    {{
        "scene1": {{
            "标题": "场景标题",
            "描述": "场景描述内容",
            "shots": [
                {{"role_id": "1", "表情和动作": "表情和动作内容", "对话": "对话内容", "type": "dialogue"}},
                {{"role_id": "3", "表情和动作": "表情和动作内容", "对话": "对话内容", "type": "dialogue"}},
                {{"role_id": "0", "风景描写": "风景内容", "type": "scenery"}},
                {{"role_id": "2", "表情和动作": "表情和动作内容", "对话": "对话内容", "type": "dialogue"}},
                {{"role_id": "1", "表情和动作": "表情和动作内容", "对话": "哈哈", "type": "action"}},
            ]
        }},
        "scene2": {{
            ...
        }}
    }}

    注意事项：
    除0以外, role_id必须从以下角色列表中选择, 确保role_id与角色的对应关系正确:
    {roles_json}
    type字段规则：
    - 对话节点 → "dialogue", role_id不为0, role_id必须在角色列表中选择
    - 动作节点（无明确对话, 强调动作、表情、交互） → "action", role_id不为0, role_id必须在角色列表中选择
    - 风景节点（role_id=0） → "scenery"
    根据剧情上下文, 在剧情中自动穿插 scenery 节点（role_id=0）, 用于风景、环境氛围等转场过渡镜头, 格式为{{"role_id": "0", "风景描写": "描写环境或景物", "type": "scenery"}}, 风景描写不能包含人物角色, 只描写环境、景物或氛围；
    根据剧情上下文, 在剧情中自动穿插 action 节点, 用于角色的动作镜头, 格式为{{"role_id": "1", "表情和动作": "表情和动作内容", "对话": "哈哈", "type": "action"}}, 对话很短, 重点描写人物动作；
    角色发言的时间先后顺序必须与原剧本一致, 同一角色可多次出现；
    表情和动作应详细具体、清晰明了, 用于指导 image2video 生成对话或动作视频；
    如果没有明确的表情和动作, 请使用"表情和动作": "在原位置保持不动"；
    dialogue和action节点的对话内容不能为空, 如果没有内容, 可以根据上下文添加常用语气词（哈、哈哈、哎、好、行等）；
    只返回JSON结构本身, 不需要额外解释。
    """
                }
        ]

        script_data = self.generate_response(messages)
        messages2 = [
            {"role": "system", "content": "你是一个剧本解析专家, 擅长解析剧情内容。"},
            {"role": "user", "content": f"""
    以下是已有的剧情信息：
    {script_data}

    请检查以下剧本, 按照时间先后顺序修正剧情中的节点顺序：

    {script_md}

    输出格式与之前一致：
    {{
        "scene1": {{
            "标题": "场景标题",
            "描述": "场景描述内容",
            "shots": [
                {{"role_id": "1", "表情和动作": "表情和动作内容", "对话": "对话内容", "type": "dialogue"}},
                {{"role_id": "3", "表情和动作": "表情和动作内容", "对话": "对话内容", "type": "dialogue"}},
                {{"role_id": "0", "风景描写": "风景内容", "type": "scenery"}},
                {{"role_id": "2", "表情和动作": "表情和动作内容", "对话": "对话内容", "type": "dialogue"}},
                {{"role_id": "1", "表情和动作": "表情和动作内容", "对话": "哈哈", "type": "action"}},
            ]
        }},
        "scene2": {{
            ...
        }}
    }}

    注意事项：
    根据剧情上下文, 按照时间先后顺序调整剧情中的节点顺序, 保持对话节点、动作节点、风景节点的时间先后顺序与原剧本一致。
    """}
        ]
        script_data_2 = self.generate_response(messages2)
        return script_data_2

    @staticmethod
    def refine_type_by_dialogue(script_data: dict):
        """
        根据对话长度修正规则：
        - 对话长度 > 3 → dialogue
        - 对话长度 <= 3 → action
        """
        for scene, scene_data in script_data.items():
            for shot in scene_data.get("shots", []):
                if shot["type"] in ["dialogue", "action"] and "对话" in shot:
                    if len(shot["对话"]) > 3:
                        shot["type"] = "dialogue"
                    else:
                        shot["type"] = "action"
        return script_data

    def generate_base_info(self):
        """
        将Markdown剧本通过LLM添加开场画面描述
        """
        path_script = self.data.base_info.path_script
        with open(path_script, "r", encoding="utf-8") as f:
            script_md = f.read()
        messages=[
            {"role": "system", "content": "你是一个剧本解析专家, 擅长将剧本内容解析为结构化的JSON格式。"},
            {"role": "user", "content": 
                f"""
    请将以下剧本内容解析为JSON格式：

    {script_md}

    要求输出结构如下（注意保留中文字段, 并包含 type 字段）：

    {{
        "base_info": {{
            "title": "整个短剧的标题",
            "description": "整个短剧的描述",
            "opening_scene": "描述开场画面, 要求画面简单, 不含有人物角色, 用于生成开场视频",
            "ending_scene": "描述结尾画面, 要求画面简单, 不含有人物角色, 用于生成结尾视频",
        }}
    }}

    注意事项：
    只返回JSON结构本身, 不需要额外解释。
    """
                }
        ]

        base_info = self.generate_response(messages)
        return base_info
    
    def generate_script_and_shot(self, skip_if_exists=True):
        """
        生成剧本和镜头
        """ 
        results=[]
        print_text=''
        # self.data = self.data.load_json()
        results.append(self.data.base_info.path_script)
        results.append(self.data.base_info.path_drama_data)
        # 如果results中文件都已存在, 则直接跳过
        if all(os.path.exists(path) for path in results) and skip_if_exists:
            print("剧本已存在, 跳过生成步骤。")
            return results, print_text
        print("[INFO] 正在生成剧本...\n")
        self.generate_script_iterative()
        roles_info = self.generate_roles_info()

        # 角色编号映射
        role_id_map = {f"{i + 1}": role for i, role in enumerate(roles_info.keys())}
        roles_dict = {}
        for role_id, role_name in role_id_map.items():
            details = roles_info[role_name]
            roles_dict[role_id] = {
                "role_id": role_id,
                "name": role_name,
                "describe": details["describe"],
                "name_en": DramaData.translate_text(role_name),
            }
        self.data.roles = roles_dict
        script_data = self.generate_json()
        # 向script_data['scene1']['shots']中添加chatID, 内容为scene1-数字, 数字为0,1,2, 3
        for scene, script_value in script_data.items():
            for index, plot in enumerate(script_value['shots']):
                plot['chatID'] = f"{scene}_{index}"
                # 如果是风景节点（type == "scenery"）
                if plot.get('type', None) == "scenery":
                    plot['name'] = "风景"
                    plot.setdefault("role_id", "0")
                    continue
                # 修正缺失的 type
                if "type" not in plot:
                    if plot.get("对话"):
                        plot["type"] = "dialogue"
                    else:
                        plot["type"] = "action"
                # 修正 role_id
                if plot['role_id'] not in self.data.roles:
                    print(f"修正角色名: {plot['role_id']} -> 1")
                    plot['role_id'] = '1'
                # 补充角色名字
                plot['name'] = self.data.roles[plot['role_id']]['name']
        # 应用 refine_type_by_dialogue
        script_data = self.refine_type_by_dialogue(script_data)
        self.data.scripts = script_data
        # 添加场景角色信息
        for scene, value in self.data.scripts.items():
            roles_id_ordered = []
            value['scene_roles'] = []
            for chat in value['shots']:
                if chat['type'] == "scenery":  # 跳过风景节点
                    continue
                chat['role_id'] = [id for id, name in self.data.roles.items() if name['name'] == chat['name']][0]
                id = chat['role_id']
                if id not in roles_id_ordered:
                    roles_id_ordered.append(id)
                    value['scene_roles'].append(
                        {
                            'role_id': f'{id}',
                            'name': chat['name'],
                        }
                    )
        base_info = self.generate_base_info()
        for key, value in base_info.get('base_info', {}).items():
            self.data.base_info[key] = value
        # 保存self.data到json文件
        self.data.save_json()
        print_text += f"✅ 剧本和镜头生成完成！\n"
        # 返回生成的剧本文件
        return results, print_text

    # 2. 生成语音
    # 调用 DeepSeek 大模型 API, 根据剧本生成角色描述和音色
    def generate_roles_and_voices(self, role_to_update=None, exclude_timbres=None):
        path_script = self.data.base_info.path_script
        with open(path_script, "r", encoding="utf-8") as f:
            script_content = f.read()
        roles_name = [roles['name'] for id, roles in self.data.roles.items()]
        # roles_name转化为字符串
        roles_name_str = ', '.join(roles_name)
        exclude_timbres = exclude_timbres or []
        while True:
            try:
                if role_to_update:
                    prompt = (
                        f"你是语音合成专家, 擅长根据剧本生成角色音色。"
                        f"避免使用以下音色：{', '.join(exclude_timbres)}。\n"
                        f"请仅为角色 {role_to_update} 选择合适的音色（使用Azure TTS音色名称, 例如 {self.shortnames_zh_voices}）。"
                        f"输出JSON格式, 例如：{{'{role_to_update}': {{'voice': '音色名称'}}}}"
                    )
                else:
                    prompt = (
                        f"你是语音合成专家, 擅长根据剧本生成角色音色。"
                        f"请仅为以下角色生成音色：{roles_name_str}（使用Azure TTS音色名称, 例如 {self.shortnames_zh_voices}）, "
                        f"每个角色使用不同音色, 输出JSON, 例如："
                        f"{{'角色1': {{'voice': '音色'}},'角色2': {{'voice': '音色'}}}}"
                    )
                
                messages = [
                    {"role": "system", "content": "你是语音合成专家。"},
                    {"role": "user", "content": f"剧本内容：\n{script_content}\n\n{prompt}"}
                ]

                roles_and_voices = self.generate_response(messages)

                if role_to_update:
                    generated_voice = roles_and_voices[role_to_update]["voice"]
                    if generated_voice not in exclude_timbres:
                        return roles_and_voices
                else:
                    # 检查是否生成的角色名都在固定列表中
                    if all(role in roles_name for role in roles_and_voices.keys()):
                        return roles_and_voices

            except Exception as e:
                print("解析角色音色失败, 重试...\n", e)
                continue

    # 合成多角色语音
    async def synthesize_roles(self, roles_and_voices):
        synthesized_roles = {}
        exclude_timbres = set()

        for role_id, role in self.data.roles.items():
            role_name = role['name']
            details = roles_and_voices[role_name]

            while role_name not in synthesized_roles:
                try:
                    print(f"正在合成角色 {role_name}的语音...\n")
                    communicate = Communicate(role["describe"], details["voice"])
                    path_voice = os.path.join(self.data.base_info.OUTPUT_DIR, f'voice_{role_id}.wav')
                    await communicate.save(path_voice)
                    print(f"{role_name} 的语音{details['voice']}合成完成！")
                    role['voice'] = details["voice"]
                    synthesized_roles[role_name] = details
                    exclude_timbres.add(details["voice"])
                except Exception as e:
                    print(f"❌ 合成角色 {role_name} 的语音时出错：{e}")
                    print(f"正在为角色 {role_name} 重新生成音色...\n")
                    exclude_timbres.add(details["voice"])
                    new_voice = self.generate_roles_and_voices(role_to_update=role_name, exclude_timbres=list(exclude_timbres))
                    if role_name in new_voice:
                        details["voice"] = new_voice[role_name]["voice"]
                        role['voice'] = details["voice"]
                    else:
                        print(f"❌ 无法为角色 {role_name} 重新生成音色, 跳过该角色。")
                        break

        return synthesized_roles

    # 合成对话音频
    async def generate_audio(self):
        for key, value in self.data.scripts.items():
            for index, plot in enumerate(value['shots']):
                role_id = plot['role_id']
                if role_id == "0":
                    continue
                chatID = plot['chatID']
                role = plot['name']
                dialogue = plot['对话']
                print(chatID)
                print(role)
                try:
                    voice = self.data.roles[role_id]['voice']
                    print(voice)
                    print(f"正在合成 {plot['chatID']} 的语音...\n")
                    communicate = Communicate(dialogue, voice)
                    path_audio = os.path.join(self.data.base_info.OUTPUT_DIR, f'{chatID}.wav')
                    await communicate.save(path_audio)
                    print(f"✅ 语音合成完成！")
                    # 添加音频路径至plot['path_audio']
                    plot['path_audio']=path_audio
                except Exception as e:
                    print(f"❌ 合成对话 {chatID} 的语音时出错：{e}")
    
    async def generate_voices(self, skip_if_exists=True):
        results=[]
        print_text=''
        self.data = self.data.load_json()
        for role_id, role in self.data.roles.items():
            path_voice = os.path.join(self.data.base_info.OUTPUT_DIR, f'voice_{role_id}.wav')
            results.append(path_voice)
        for key, value in self.data.scripts.items():
            for index, plot in enumerate(value['shots']):
                if plot['type'] != 'scenery':
                    chatID = plot['chatID']
                    path_audio = os.path.join(self.data.base_info.OUTPUT_DIR, f'{chatID}.wav')
                    results.append(path_audio)
        # 如果results中所有文件都存在, 则直接跳过
        if all(os.path.exists(path) for path in results) and skip_if_exists:
            print("所有语音已存在, 跳过生成步骤。")
            return results, print_text
        print("[INFO] 正在生成语音...\n")
        voices = await edge_tts.list_voices()
        # 获取所有支持中文的语音
        zh_voices = [voice for voice in voices if "zh" in voice["Locale"]]
        # 获取short name列表
        shortnames_zh_voices = [voice["ShortName"] for voice in zh_voices]
        # print(shortnames_zh_voices)
        self.shortnames_zh_voices = shortnames_zh_voices
        # 根据剧本生成角色描述和音色
        roles_and_voices = self.generate_roles_and_voices()
        # 合成语音（传入编号映射）
        roles_and_voices = await self.synthesize_roles(roles_and_voices)
        await self.generate_audio()
        # 保存self.data到json文件
        self.data.save_json()
        print_text += f"✅ 语音生成完成！\n"
        return results, print_text
            
    # def generate_voices_sync(self):
    #     import asyncio
    #     return asyncio.get_event_loop().run_until_complete(self.generate_voices())
    def generate_voices_sync(self, skip_if_exists=True):
        return asyncio.run(self.generate_voices(skip_if_exists=skip_if_exists))

    # 3. 生成角色
    def generate_roles(self, skip_if_exists=True):
        """调用 LLM 生成角色提示词, 并生成角色图像"""
        results=[]
        print_text=''
        self.data = self.data.load_json()
        for role_id in self.data.roles.keys():
            path_image = os.path.join(self.data.base_info.OUTPUT_DIR, f"{role_id}.png")
            results.append(path_image)
        # 如果results中文件都已存在, 则直接跳过
        if all(os.path.exists(path) for path in results) and skip_if_exists:
            print("所有角色已存在, 跳过生成步骤。\n")
            return results, print_text
        
        print("[INFO] 正在生成角色...\n")
        prompts_init, prompts_update = self.generate_roles_prompts()
        for role_id, role_info in self.data.roles.items():
            if role_id in prompts_init:
                # 初始提示词
                self.data.roles[role_id]["prompt"] = prompts_init[role_id]["正面提示词"]
                self.data.roles[role_id]["negative_prompt"] = prompts_init[role_id]["负面提示词"]
                self.data.roles[role_id]["prompt_en"] = DramaData.translate_text(prompts_init[role_id]["正面提示词"])
                self.data.roles[role_id]["negative_prompt_en"] = DramaData.translate_text(prompts_init[role_id]["负面提示词"])

            if role_id in prompts_update:
                # 优化后的提示词
                self.data.roles[role_id]["update_prompt"] = prompts_update[role_id]["正面提示词优化"]
                self.data.roles[role_id]["update_negative_prompt"] = prompts_update[role_id]["负面提示词优化"]
                self.data.roles[role_id]["update_prompt_en"] = DramaData.translate_text(prompts_update[role_id]["正面提示词优化"])
                self.data.roles[role_id]["update_negative_prompt_en"] = DramaData.translate_text(prompts_update[role_id]["负面提示词优化"])
        self.generate_roles_image(skip_if_exists=skip_if_exists)
        for role_id in self.data.roles.keys():
            path_image = os.path.join(self.data.base_info.OUTPUT_DIR, f"{role_id}.png")
            self.extract_person(
                input_path=path_image,
                output_path=path_image.replace('.png', '_clean.png')
            )
        # 保存self.data到json文件
        self.data.save_json()
        print_text += f"✅ 角色生成完成！\n"
        return results, print_text

    def generate_roles_prompts(self):
        path_script = self.data.base_info.path_script
        roles_info = {scene: value['name'] for scene, value in self.data.roles.items()}
        with open(path_script, "r", encoding="utf-8") as f:
            script_content = f.read()

        # Step 1: 生成初始提示词
        messages_init = [
            {"role": "system", "content": "你是一个视觉提示词专家, 擅长为AI绘画生成角色形象图的正面和负面提示词, 用于生成清晰完整美观的角色形象图。"},
            {"role": "user", "content": f"""
    剧本语言为{self.data.base_info.language}, 短剧风格为：{self.data.base_info.drama_style}, 文化背景为：{self.data.base_info.culture_background}, 
    剧本内容为：\n{script_content}\n, 请为以下角色 {roles_info} 生成AI绘画的正面提示词和负面提示词, 
    正面提示词用于详细描述角色形象, 包括：
    - 面容（脸型、肤色、表情）
    - 五官（眼睛、鼻子、嘴巴、耳朵）
    - 头发（颜色、长度、发型）
    - 体形（身材胖瘦、年龄特征）
    - 服饰（颜色、材质、样式、装饰）
    - 必要的配饰或道具
    背景为纯白色空白, 脚部要穿鞋或遮挡, 上身要穿衣服, 严禁裸体或露出脚趾。
    负面提示词用于排除不需要的元素, 包括：
    模糊、畸形、低质量、不符合文化背景{self.data.base_info.culture_background}或短剧风格{self.data.base_info.drama_style}、
    面部特征缺失、面貌丑陋或恐怖、身体畸形、眼睛畸形或不对称、鼻子或嘴巴模糊、手指数量错误、服饰缺失、
    身体缺失、裸体、上身赤裸、露出脚趾、背景杂乱、背景有物体等。
    输出格式为JSON, 例如：{{'role_id': {{'正面提示词': '...', '负面提示词': '...'}}, ...}}。role_id为角色id, 比如1,2,3,4,...。

    注意事项：
        - 提示词内容不得超过180个字；
        - 角色必须高清, 身体必须完整, 身体比例协调, 不能缺失身体、四肢等；
        - 角色形象在符合剧本内容的前提下, 尽量美观友好, 不要恐怖或丑陋；
        - 手指必须正常自然, 避免手指畸形或手指数量不对；
        - 人物上身要穿衣服, 脚部要穿鞋或遮挡, 严禁裸体或露出脚趾；
        - 要求直观详细描述, 不要用缩写或成语等不直观的表达；
        - 提示词中不要出现人物名称等不直观的特有名词；
        - 对所有物品（武器、饰品、服饰、道具等）必须使用最直观的说明来描述特征、形状、颜色、尺寸大小等特征, 
            不允许直接使用含文化背景的专有名词, 例如：
            “和尚” 应写成 “穿着中国僧服的没有头发的男性”, 
            “仙女” 应写成 “穿着浅色的中国古代服饰的纯洁可爱的年轻美丽女性”, 
            “女妖精” 应写成 “穿着中国古代服饰的妖娆性感的年轻美丽女性”, 
            “金箍棒” 应写成 “金黄色的长度约1.8米的圆柱形铁棒”, 
            “绣花鞋” 应写成 “有着花纹图案的美丽布鞋”, 
            “青龙偃月刀” 应写成 “带有长手柄的长2m的青铜色金属大刀”, 
            “禅杖” 应写成 “带有金属光泽的长圆柱体, 顶部为多个圆环组成的规则球形物体”, 
            要确保没有依赖先验文化知识的专有词汇, 所有描述都应该通俗易懂、清楚直观。
            """}
        ]
        prompts_init = self.generate_response(messages_init)

        # Step 2: 基于初始提示词, 生成优化后的提示词
        messages_update = [
            {"role": "system", "content": "你是一个视觉提示词优化专家, 能够在初始提示词基础上优化, 使角色更符合剧情、更美观、符合文化背景与短剧风格。"},
            {"role": "user", "content": f"""
                以下是剧本的设定：
                - 剧本语言: {self.data.base_info.language}
                - 短剧风格: {self.data.base_info.drama_style}
                - 文化背景: {self.data.base_info.culture_background}
                - 剧本内容: \n{script_content}\n

                以下是每个角色的初始提示词:
                {prompts_init}

                请在这些初始提示词的基础上, 为每个角色生成优化后的正面提示词和负面提示词, 
                目标是让角色更符合剧情内容, 符合文化背景和短剧风格, 保持纯白背景, 不要出现人物名称等不直观的特有名词。
                避免模糊和抽象的描述, 人物身体必须完整, 手指必须自然正常, 不能缺失身体、四肢等。
                要求输出 JSON 格式, 例如：
                {{'role_id': {{'正面提示词优化': '...', '负面提示词优化': '...'}}, ...}}。role_id为角色id, 比如1,2,3,4,...。

            """}
        ]
        prompts_update = self.generate_response(messages_update)
        return prompts_init, prompts_update

    def generate_roles_image(self, skip_if_exists=True, max_length=300):
        conda_env = "fast_video"
        uvicorn_host = "127.0.0.1"
        uvicorn_port = 5101
        api_url = f"http://{uvicorn_host}:{uvicorn_port}/text2image"
        model_api = f"http://{uvicorn_host}:{uvicorn_port}/load_model"

        # ------------------- 启动 FastAPI -------------------
        fastapi_process = subprocess.Popen(
            [
                "conda", "run", "-n", conda_env,
                "uvicorn", "fastapi_text2image:app",
                "--host", uvicorn_host,
                "--port", str(uvicorn_port)
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"FastAPI 已启动 (虚拟环境: {conda_env}), 等待模型加载...\n")

        # 等待服务就绪
        self.wait_for_fastapi_ready(api_url)

        # 加载指定模型
        print(f"请求 FastAPI 加载模型: {self.data.base_info.text2image_model}")
        resp = requests.post(model_api, json={"model_id": self.data.base_info.text2image_model})
        resp.raise_for_status()
        print(resp.json())

        try:
            roles_info = self.data.roles
            for role_id, info in roles_info.items():
                final_path = os.path.join(self.data.base_info.OUTPUT_DIR, f"{role_id}.png")
                # 如果存在则跳过
                if skip_if_exists and os.path.exists(final_path):
                    print(f"角色 {role_id}: {info['name']} 已存在, 跳过生成。\n")
                    continue
                # 存放候选图的分数
                candidate_scores = []
                for i in range(4):
                    if i%2==0:
                        prompt = f"High-definition picture quality.{info['prompt_en']}.{self.data.base_info.drama_style}.{self.data.base_info.culture_background}"
                        negative_prompt = info["negative_prompt_en"]
                    else:
                        prompt = f"High-definition picture quality.{info['update_prompt_en']}.{self.data.base_info.drama_style}.{self.data.base_info.culture_background}"
                        negative_prompt = info["update_negative_prompt_en"]
                    prompt = ' '.join(prompt.split()[:max_length])
                    negative_prompt = ' '.join(negative_prompt.split()[:max_length])
                    print(f"生成 {role_id}: {info['name']} 的形象")
                    print(f"正面提示词: {prompt}")
                    print(f"负面提示词: {negative_prompt}")
                    # 调用 FastAPI 生成图像
                    image = self.text2image_via_api(
                        prompt, negative_prompt,
                        width=self.data.base_info.width,
                        height=self.data.base_info.height,
                        api_url=api_url,
                        steps=40
                    )
                    # 转化image为统一的长宽
                    path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{role_id}_{i}.png")
                    resized_img = self.resize_keep_aspect(image=image, save_path=path_output)
                    # display(resized_img)
                    del image, resized_img
                    score, comps = self.score_character(path_output, prompt)
                    candidate_scores.append((score, path_output))
                    print(f"候选角色 {role_id}_{i}.png 分数: {score}/100, 分项: {comps}")
                    gc.collect()
                    torch.cuda.empty_cache()
                # 选出分数最高的
                best_score, best_path = max(candidate_scores, key=lambda x: x[0])
                shutil.copy(best_path, final_path)
                print(f"角色{role_id} 最佳图: {os.path.basename(best_path)} (score={best_score}) 已保存为 {final_path}")

        finally:
            print("请求 FastAPI 自我关闭, 释放显存...\n")
            # 等待所有请求完成
            while True:
                try:
                    resp = requests.get(f"http://{uvicorn_host}:{uvicorn_port}/busy", timeout=20)
                    if resp.status_code == 200 and not resp.json().get("busy", False):
                        break
                except Exception:
                    break
                print("FastAPI 正在推理, 等待完成再关闭...\n")
                time.sleep(20)

            try:
                requests.post(f"http://{uvicorn_host}:{uvicorn_port}/shutdown", timeout=5)
            except Exception as e:
                print("FastAPI 已经关闭或无法访问:", e)

            try:
                fastapi_process.wait(timeout=10)
                print("✅ FastAPI 已关闭")
            except Exception:
                print("FastAPI 未能在规定时间内退出, 执行强制终止...\n")
                fastapi_process.terminate()
                try:
                    fastapi_process.wait(timeout=5)
                except Exception:
                    print("❌ FastAPI 无法正常终止, 执行强制 kill")
                    fastapi_process.kill()
                print("FastAPI 已被强制关闭")

    # 4. 生成场景
    def generate_scenes(self, skip_if_exists=True):
        """根据剧本生成场景图像"""
        results=[]
        print_text=''
        self.data = self.data.load_json()
        for text in ['opening', 'ending']:
            path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{text}_background.png")
            results.append(path_output)
        results += [(os.path.join(self.data.base_info.OUTPUT_DIR, "opening_scene.png"))]
        for scene, value in self.data.scripts.items():
            path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{scene}_opening_background.png")
            results.append(path_output)
        for scene, value in self.data.scripts.items():
            for scene_role in value['scene_roles']:
                role_id = scene_role['role_id']
                final_path = os.path.join(self.data.base_info.OUTPUT_DIR, f"{scene}_role_{role_id}.png")
                results.append(final_path)
        for scene, value in self.data.scripts.items():
            for idx, chat_info in enumerate(value['shots']):
                if str(chat_info['type']) == "scenery":
                    chatID = chat_info['chatID']
                    path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}_scenery.png")
                    results.append(path_output)
        # 如果results中文件都已存在, 则直接跳过
        if all(os.path.exists(path) for path in results) and skip_if_exists:
            print("所有场景已存在, 跳过生成步骤。\n")
            return results, print_text
        
        print("[INFO] 正在生成场景...\n")
        background_prompts = self.generate_scene_background_prompts()
        for scene, value in self.data.scripts.items():
            value['background_prompt'] = background_prompts[scene]['正面提示词']
            value['background_negative_prompt'] = background_prompts[scene]['负面提示词']
            value['background_prompt_en'] = DramaData.translate_text(value['background_prompt'])
            value['background_negative_prompt_en'] = DramaData.translate_text(value['background_negative_prompt'])
        
        path_background=self.generate_scene_background_image(skip_if_exists=skip_if_exists)
        # results.append(path_background)
        if skip_if_exists and os.path.exists(os.path.join(self.data.base_info.OUTPUT_DIR, "opening_scene.png")):
            print("opening_scene 已存在, 跳过生成。\n")
        else:
            # 更鲁棒的合成：按比例缩放（保持长宽比）、使用 alpha 掩码、最多合成前 2 个角色、容错缺失文件
            opening_image = Image.open(os.path.join(self.data.base_info.OUTPUT_DIR, "opening_background.png")).convert("RGBA")

            # 收集最多两个角色图（根据 drama_data 中的角色 id 顺序）
            role_ids = list(self.data.roles.keys())[:2]
            role_paths = [os.path.join(self.data.base_info.OUTPUT_DIR, f"{rid}_clean.png") for rid in role_ids]
            role_paths = [p for p in role_paths if os.path.exists(p)]

            if not role_paths:
                print("未找到角色图, 保存原背景")
                opening_image.save(os.path.join(self.data.base_info.OUTPUT_DIR, "opening_scene.png"))
                # display(opening_image)
            else:
                bg = opening_image.copy()
                # 目标尺寸：单人时占比更大, 双人时为双方留出空间
                if len(role_paths) == 1:
                    max_w = int(bg.width * 0.5)
                    max_h = int(bg.height * 0.6)
                else:
                    max_w = int(bg.width * 0.45)
                    max_h = int(bg.height * 0.6)

                y = int(bg.height * 0.3)  # 垂直位置基准

                # 打开并按容器尺寸等比缩放
                imgs = []
                for p in role_paths:
                    im = Image.open(p).convert("RGBA")
                    im = ImageOps.contain(im, (max_w, max_h))  # 保持长宽比缩放到框内
                    imgs.append(im)

                # 计算水平位置并粘贴（使用自身 alpha 作为 mask）
                if len(imgs) == 1:
                    im = imgs[0]
                    x = (bg.width - im.width) // 2
                    bg.paste(im, (x, y), im)
                else:
                    left_x = int(bg.width * 0.25 - imgs[0].width // 2)
                    right_x = int(bg.width * 0.75 - imgs[1].width // 2)
                    bg.paste(imgs[0], (left_x, y), imgs[0])
                    bg.paste(imgs[1], (right_x, y), imgs[1])

                # display(bg)
                bg.save(os.path.join(self.data.base_info.OUTPUT_DIR, "opening_scene.png"))
        scene_role_prompts = self.generate_scene_role_prompts()
        for scene, value in self.data.scripts.items():
            for role in value['scene_roles']:
                # 人物角色 prompt
                role['prompt'] = scene_role_prompts[scene][role['role_id']]['正面提示词']
                role['negative_prompt'] = scene_role_prompts[scene][role['role_id']]['负面提示词']
                role['prompt_en'] = DramaData.translate_text(role['prompt'])
                role['negative_prompt_en'] = DramaData.translate_text(role['negative_prompt'])
        path_scene_roles=self.generate_scene_role_image(skip_if_exists=skip_if_exists)
        # results += path_scene_roles
        scenery_prompts = self.generate_scenery_prompts()

        for scene, value in self.data.scripts.items():
            for chat in value['shots']:
                if chat['type'] == "scenery":
                    # scenery 节点 prompt 使用 chatID
                    chat_id = chat['chatID']
                    chat['prompt'] = scenery_prompts[scene][chat_id]['正面提示词']
                    chat['negative_prompt'] = scenery_prompts[scene][chat_id]['负面提示词']
                    chat['prompt_en'] = DramaData.translate_text(chat['prompt'])
                    chat['negative_prompt_en'] = DramaData.translate_text(chat['negative_prompt'])
        path_scene_scenery=self.generate_scene_scenery_image(skip_if_exists=skip_if_exists)
        # results += path_scene_scenery
        # 保存self.data到json文件
        self.data.save_json()
        print_text += f"✅ 场景生成完成！\n"
        return results, print_text

    # 生成每个场景背景的提示词, 用于text2image生成场景背景图
    def generate_scene_background_prompts(self):
        scripts=self.data.scripts
        messages=[
                {"role": "system", "content": f"你是一个视觉提示词专家, 擅长为AI绘画生成短剧或电影的场景背景图的正面和负面提示词,注意不含有人物角色。"},
                {"role": "user", "content": 
                f'''
                剧本语言为{self.data.base_info.language}, 短剧风格为：{self.data.base_info.drama_style}, 文化背景为：{self.data.base_info.culture_background}, 
                    剧本内容为：\n{scripts}\n, 请为所有场景 {', '.join(scripts)} 生成AI绘画的正面提示词和负面提示词, 
                    正面提示词用于详细描述场景背景的内容, 如地点、背景、光照等, 不含有人物角色。 
                    负面提示词用于排除不需要的元素, 如模糊、畸形、低质量、不符合 {self.data.base_info.culture_background}、不符合 {self.data.base_info.drama_style}等。
                    输出格式为JSON, 例如：{{'scene1': {{'正面提示词': '...', '负面提示词': '...'}}, ...}}。
                    注意事项：
                    - 提示词内容为{self.data.base_info.language},不得超过100个字；
                    - 所有元素必须符合{self.data.base_info.drama_style} {self.data.base_info.culture_background}；
                    - 要求直观详细描述, 不要用缩写或成语等不直观的表达；
                    - 对所有物品（武器、饰品、服饰、道具等）必须使用最直观的说明来描述形状、颜色、尺寸等特征, 
                    不允许直接使用含文化背景的专有名词, 例如：
                    “参天大树” 应写成 “十分高大粗壮的树”, 
                    “鹅毛大雪” 应写成 “雪花很大的大雪”, 
                    “七彩祥云” 应写成 “彩色的云朵”, 
                    总之要确保没有依赖先验文化知识的专有词汇, 让描述能让从未听过该名词的人直接理解画面。
                '''
                }
            ]
        prompts = self.generate_response(messages)
        return prompts

    def generate_scene_background_image(self, skip_if_exists=True, max_length=500):
        ''' 
        :param scene_roles_id: 指定生成哪些角色, 比如 ["scene1_role_2", "scene2_role_3"]
        '''
        conda_env = "fast_video"
        uvicorn_host = "127.0.0.1"
        uvicorn_port = 5102
        api_url = f"http://{uvicorn_host}:{uvicorn_port}/text2image"
        model_api = f"http://{uvicorn_host}:{uvicorn_port}/load_model"
        results = []
        # ------------------- 启动 FastAPI -------------------
        fastapi_process = subprocess.Popen(
            [
                "conda", "run", "-n", conda_env,
                "uvicorn", "fastapi_text2image:app",
                "--host", uvicorn_host,
                "--port", str(uvicorn_port)
            ],
            # stdout=subprocess.PIPE, stderr=subprocess.PIPE
            stdout= None,
            stderr= None
        )
        print(f"FastAPI 已启动 (虚拟环境: {conda_env}), 等待模型加载...\n")
        # 等待服务就绪
        self.wait_for_fastapi_ready(api_url)
        # 加载指定模型
        print(f"请求 FastAPI 加载模型: {self.data.base_info.text2image_model}")
        resp = requests.post(model_api, json={"model_id": self.data.base_info.text2image_model})
        resp.raise_for_status()
        print(resp.json())
        try:
            for text in ['opening', 'ending']:
                path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{text}_background.png")
                # 如果存在则跳过
                if skip_if_exists and os.path.exists(path_output):
                    print(f"{text}_background 已存在, 跳过生成。\n")
                    continue
                print(f"{text}_background")
                prompt = self.data.base_info[f'{text}_scene'] + self.data.base_info.drama_style + self.data.base_info.culture_background
                prompt = DramaData.translate_text(prompt)
                prompt = f'''
        High-definition picture quality. {prompt}
                '''
                prompt = " ".join(prompt.split()[:max_length])
                negative_prompt = f'''
        Blurry, low quality, person, human, character, character shadows, figure, silhouette, man, woman, people, portrait, face, body, text, watermark, logo
                '''
                negative_prompt = " ".join(negative_prompt.split()[:max_length])
                print('prompt:\n', prompt)
                print('negative_prompt:\n', negative_prompt)
                # 调用 FastAPI 生成图像
                image = self.text2image_via_api(
                    prompt=prompt, 
                    negative_prompt=negative_prompt,
                    width=self.data.base_info.width,
                    height=self.data.base_info.height,
                    api_url=api_url,
                    steps=40
                )

                # 转化image为统一的长宽
                resized_img = self.resize_keep_aspect(image=image, save_path=path_output)
                # role["path_role"] = path_role
                results.append(path_output)
                # display(resized_img)
                del image, resized_img
                gc.collect()
                torch.cuda.empty_cache()
            for scene, value in self.data.scripts.items():
                print(scene)
                path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{scene}_opening_background.png")
                # 如果存在则跳过
                if skip_if_exists and os.path.exists(path_output):
                    print(f"{scene}_opening_background 已存在, 跳过生成。\n")
                    continue
                prompt = f'''
    High-definition picture quality. {value['background_prompt_en']}. {self.data.base_info.drama_style}.{self.data.base_info.culture_background}
                '''
                prompt = " ".join(prompt.split()[:max_length])
                negative_prompt = f'''
    Blurry, low quality, person, human, character, character shadows, figure, silhouette, man, woman, people, portrait, face, body, text, watermark, logo.
                '''
                negative_prompt = " ".join(negative_prompt.split()[:max_length])
                print('prompt:\n', prompt)
                print('negative_prompt:\n', negative_prompt)
                # 调用 FastAPI 生成图像
                image = self.text2image_via_api(
                    prompt=prompt, 
                    negative_prompt=negative_prompt,
                    width=self.data.base_info.width,
                    height=self.data.base_info.height,
                    api_url=api_url,
                    steps=40
                )

                # 转化image为统一的长宽
                resized_img = self.resize_keep_aspect(image=image, save_path=path_output)
                results.append(path_output)
                # role["path_role"] = path_role
                # display(resized_img)
                del image, resized_img
                gc.collect()
                torch.cuda.empty_cache()

        finally:
            print("请求 FastAPI 自我关闭, 释放显存...\n")
            # 等待所有请求完成
            while True:
                try:
                    resp = requests.get(f"http://{uvicorn_host}:{uvicorn_port}/busy", timeout=20)
                    if resp.status_code == 200 and not resp.json().get("busy", False):
                        break
                except Exception:
                    break
                print("FastAPI 正在推理, 等待完成再关闭...\n")
                time.sleep(20)

            try:
                requests.post(f"http://{uvicorn_host}:{uvicorn_port}/shutdown", timeout=5)
            except Exception as e:
                print("FastAPI 已经关闭或无法访问:", e)

            try:
                fastapi_process.wait(timeout=10)
                print("✅ FastAPI 已关闭")
            except Exception:
                print("FastAPI 未能在规定时间内退出, 执行强制终止...\n")
                fastapi_process.terminate()
                try:
                    fastapi_process.wait(timeout=5)
                except Exception:
                    print("❌ FastAPI 无法正常终止, 执行强制 kill")
                    fastapi_process.kill()
                print("FastAPI 已被强制关闭")
        return results
    
    # 生成每个场景的场景角色图的提示词, 用于image2image生成场景角色图
    def generate_scene_role_prompts(self):
        scripts=self.data.scripts
        all_scene_roles = {scene: value['scene_roles'] for scene, value in scripts.items()}
        messages = [
                {"role": "system", "content": f"你是一个视觉提示词专家, 擅长为AI绘画image2image生成正面和负面提示词, 描述角色的表情、动作、姿势等特征, 保持角色的外貌、服装不变。"},
                {"role": "user", "content": 
                f'''
                剧本语言为{self.data.base_info.language}, 短剧风格为：{self.data.base_info.drama_style}, 文化背景为：{self.data.base_info.culture_background}, 
                    剧本内容为：\n{scripts}\n, 请为所有场景 {scripts.keys()} 生成对应的人物角色{all_scene_roles}的AI绘画的正面提示词和负面提示词, 
                    正面提示词用于详细描述当前场景下的当前角色的表情、动作、姿势等特征以及环境背景, 无需描写角色的外貌、服装等特征（要求外貌、服装与输入图片中的角色一致）, 角色形象必须清晰和完整。
                    负面提示词用于排除不需要的元素, 如模糊、畸形、手指畸形、低质量、不符合文化背景{self.data.base_info.culture_background}或短剧风格{self.data.base_info.drama_style} 、面貌丑陋、身体缺失、裸体、露出脚趾等。
                    输出格式为JSON, 例如：{{'scene1':{{'role_id': {{'正面提示词': '...', '负面提示词': '...'}}, ...}}, 'scene2':{{'role_id': {{'正面提示词': '...', '负面提示词': '...'}}, ...}}, ...}}, role_id为对应的角色id, 例如1,2,3,4,...。
                    注意事项：
                    - 提示词内容为{self.data.base_info.language},不得超过100个字；
                    - 保持当前角色的外貌和服饰不变, 根据当前上下文调整角色的表情、动作、姿势；
                    - 环境背景要符合剧本上下文, 同一个场景中的环境背景要有相同的风格；
                    - 人物角色的身体要完整, 五个手指要正常自然, 脚部要穿鞋或隐藏, 严禁裸体或露出脚趾；
                    - 必须是一个角色, 不能出现多个角色；
                    - 提示词中不要出现角色名字, 只需要详细客观描述角色特征；
                    - 要求详细直观描述, 不要用缩写或成语等不直观的表达；
                    - 对所有物品（武器、饰品、服饰、道具等）必须使用最直观的说明来描述形状、颜色、尺寸等特征, 
                    不允许直接使用含文化背景的专有名词, 例如：
                    “和尚” 应写成 “穿着中国僧服的没有头发的男性”, 
                    “仙女” 应写成 “穿着浅色的中国古代服饰的纯洁可爱的年轻美丽女性”, 
                    “女妖精” 应写成 “穿着中国古代服饰的妖娆性感的年轻美丽女性”, 
                    “金箍棒” 应写成 “金黄色的长度约1.8米的圆柱形铁棒”, 
                    “玉净瓶” 应写成 “白色的纤细陶瓷瓶”, 
                    “绣花鞋” 应写成 “有着花纹图案的美丽布鞋”, 
                    总之要确保没有依赖先验文化知识的专有词汇, 让描述能让从未听过该名词的人直接理解画面。
                '''
                }
            ]
        result = self.generate_response(messages)
        return result

    def generate_scene_role_image(self, max_length=500, skip_if_exists=True):
        ''' 
        :param skip_if_exists: 是否跳过已有文件
        '''
        conda_env = "fast_video"
        uvicorn_host = "127.0.0.1"
        uvicorn_port = 5111
        api_url = f"http://{uvicorn_host}:{uvicorn_port}/image2image"
        model_api = f"http://{uvicorn_host}:{uvicorn_port}/load_model"
        results = []
        # ------------------- 启动 FastAPI -------------------
        fastapi_process = subprocess.Popen(
            [
                "conda", "run", "-n", conda_env,
                "uvicorn", "fastapi_image2image:app",
                "--host", uvicorn_host,
                "--port", str(uvicorn_port)
            ],
            # stdout=subprocess.PIPE, stderr=subprocess.PIPE
            stdout= None,
            stderr= None
        )
        print(f"FastAPI 已启动 (虚拟环境: {conda_env}), 等待模型加载...\n")

        # 等待服务就绪
        self.wait_for_fastapi_ready(api_url)

        # 加载指定模型
        print(f"请求 FastAPI 加载模型: {self.data.base_info.image2image_model}")
        resp = requests.post(model_api, json={"model_id": self.data.base_info.image2image_model})
        resp.raise_for_status()
        print(resp.json())

        def _generate_role_image(path_input_image, prompt, negative_prompt, path_output_image):
            """生成单个图像"""
            init_image = Image.open(path_input_image)
            image_b64 = self.pil_to_base64(init_image)
            prompt = ' '.join(prompt.split()[:max_length])
            negative_prompt = ' '.join(negative_prompt.split()[:max_length])
            print(f"正面提示词: {prompt}")
            print(f"负面提示词: {negative_prompt}")
            # 调用 FastAPI 生成图像
            image = self.image2image_via_api(
                prompt=prompt, 
                image_base64=image_b64,
                negative_prompt=negative_prompt,
                width=self.data.base_info.width,
                height=self.data.base_info.height,
                api_url=api_url,
                steps=30
            )
            # 转化image为统一的长宽
            resized_img = self.resize_keep_aspect(image=image, save_path=path_output_image)
            # display(resized_img)
            del image, resized_img
            gc.collect()
            torch.cuda.empty_cache()
        try:
            # if scene_roles_id is None:
            for scene, value in self.data.scripts.items():
                print(scene)
                for scene_role in value['scene_roles']:
                    role_id = scene_role['role_id']
                    final_path = os.path.join(self.data.base_info.OUTPUT_DIR, f"{scene}_role_{role_id}.png")
                    # 如果文件已存在, 跳过
                    if skip_if_exists and os.path.exists(final_path):
                        print(f"{scene}_role_{role_id} 已存在, 跳过生成。\n")
                        continue
                    # 加载初始角色形象图
                    path_input = os.path.join(self.data.base_info.OUTPUT_DIR, f"{role_id}.png")
                    prompt = f'''High-definition picture quality.Keep the character's appearance and clothing unchanged. 
                {scene_role['prompt_en']}. {value['background_prompt_en']}'''
                    negative_prompt = scene_role["negative_prompt_en"]
                    print(f"生成 {scene}_role_{role_id}: {scene_role['name']} 的形象")
                    # 存放候选图的分数
                    candidate_scores = []
                    for i in range(2):
                        path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{scene}_role_{role_id}_{i}.png")
                        _generate_role_image(
                            path_input_image=path_input,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            path_output_image=path_output,
                            )
                        score, comps = self.score_character(path_output, prompt)
                        candidate_scores.append((score, path_output))
                        print(f"候选 {scene}_role_{role_id}_{i}.png 分数: {score}/100, 分项: {comps}")
                        gc.collect()
                        torch.cuda.empty_cache()
                    # 选出分数最高的
                    best_score, best_path = max(candidate_scores, key=lambda x: x[0])
                    results.append(final_path)
                    shutil.copy(best_path, final_path)
                    print(f"{scene}_role_{role_id} 最佳图: {os.path.basename(best_path)} (score={best_score}) 已保存为 {final_path}")

            # else:
            #     for x in scene_roles_id:
            #         scene, role_id = x.split('_role_')
            #         scene_role = next(
            #             r for r in self.data.scripts[scene]['scene_roles']
            #             if str(r['role_id']) == str(role_id)
            #         )
            #         role_id = scene_role['role_id']
            #         # 加载初始角色形象图
            #         path_input = os.path.join(self.data.base_info.OUTPUT_DIR, f"{role_id}.png")
            #         prompt = f'''High-definition picture quality.Keep the character's appearance and clothing unchanged. 
            #     {scene_role['prompt_en']}. {self.data.scripts[scene]['background_prompt_en']}'''
            #         negative_prompt = scene_role["negative_prompt_en"]
            #         print(f"生成 {scene}_role_{role_id}: {scene_role['name']} 的形象")
            #         path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{scene}_role_{role_id}.png")
            #         _generate_role_image(
            #             path_input_image=path_input,
            #             prompt=prompt,
            #             negative_prompt=negative_prompt,
            #             path_output_image=path_output,
            #         )
            #         results.append(path_output)
        finally:
            print("请求 FastAPI 自我关闭, 释放显存...\n")
            # 等待所有请求完成
            while True:
                try:
                    resp = requests.get(f"http://{uvicorn_host}:{uvicorn_port}/busy", timeout=200)
                    if resp.status_code == 200 and not resp.json().get("busy", False):
                        break
                except Exception:
                    break
                print("FastAPI 正在推理, 等待完成再关闭...\n")
                time.sleep(20)

            try:
                requests.post(f"http://{uvicorn_host}:{uvicorn_port}/shutdown", timeout=5)
            except Exception as e:
                print("FastAPI 已经关闭或无法访问:", e)

            try:
                fastapi_process.wait(timeout=10)
                print("✅ FastAPI 已关闭")
            except Exception:
                print("FastAPI 未能在规定时间内退出, 执行强制终止...\n")
                fastapi_process.terminate()
                try:
                    fastapi_process.wait(timeout=5)
                except Exception:
                    print("❌ FastAPI 无法正常终止, 执行强制 kill")
                    fastapi_process.kill()
                print("FastAPI 已被强制关闭")
        return results

    # 生成每个场景中 scenery 节点的提示词, JSON 使用 chatID
    def generate_scenery_prompts(self):
        scripts = self.data.scripts
        # 提取所有风景节点的 chatID
        scenery_nodes = {
            scene: {chat["chatID"]: chat for chat in value["shots"] if chat["type"] == "scenery"}
            for scene, value in scripts.items()
        }
        messages = [
            {"role": "system", "content": f"你是一个视觉提示词专家, 擅长为AI绘画image2image生成正面和负面提示词, 用于生成转场的风景图像, 短剧风格为：{self.data.base_info.drama_style}, 文化背景为：{self.data.base_info.culture_background}"},
            {"role": "user", "content": 
            f'''
    剧本语言为{self.data.base_info.language}, 短剧风格为：{self.data.base_info.drama_style}, 文化背景为：{self.data.base_info.culture_background},
    剧本内容为：\n{scenery_nodes}\n,
    请为这些风景节点生成正面提示词和负面提示词, 要求：
    1. 正面提示词需详细描述风景, 包括：
    - 场景环境、建筑、自然元素
    - 光线氛围（时间、方向、色温）
    - 镜头选择：局部/特写、角度、位置、推拉/平移等镜头动作
    - 景物变化（季节、时间、动态元素如雨雪雾）
    - 保持画面清晰、真实自然、电影级效果
    2. 负面提示词需排除不需要的元素, 例如：
    - 人物、模糊、畸形、低质量、奇怪物体、违背文化背景的内容
    3. 输出格式为 JSON, 结构如下：
    {{
        "scene1": {{
            "chatID1": {{
                "正面提示词": "...",
                "负面提示词": "..."
            }},
            "chatID2": {{ ... }}
        }},
        "scene2": {{ ... }}
    }}
    其中 chatID 是节点的唯一标识符, 例如 scene1_0, scene2_1。
    注意事项：
    - 提示词不得超过100个字；
    - 必须是风景画面, 不允许出现人物；
    - 要求详细直观描述, 不能使用缩写或成语。
    - 你可以自行决定是否生成局部特写、镜头角度、镜头移动或光线变化
            '''
            }
        ]
        result = self.generate_response(messages)
        return result
        
    def generate_scene_scenery_image(self, max_length=500, skip_if_exists=True):
        ''' 
        :param skip_if_exists: 是否跳过已经存在的图像
        '''
        conda_env = "fast_video"
        uvicorn_host = "127.0.0.1"
        uvicorn_port = 5112
        api_url = f"http://{uvicorn_host}:{uvicorn_port}/image2image"
        model_api = f"http://{uvicorn_host}:{uvicorn_port}/load_model"
        results = []
        # ------------------- 启动 FastAPI -------------------
        fastapi_process = subprocess.Popen(
            [
                "conda", "run", "-n", conda_env,
                "uvicorn", "fastapi_image2image:app",
                "--host", uvicorn_host,
                "--port", str(uvicorn_port)
            ],
            # stdout=subprocess.PIPE, stderr=subprocess.PIPE
            stdout= None,
            stderr= None
        )
        print(f"FastAPI 已启动 (虚拟环境: {conda_env}), 等待模型加载...\n")

        # 等待服务就绪
        self.wait_for_fastapi_ready(api_url)

        # 加载指定模型
        print(f"请求 FastAPI 加载模型: {self.data.base_info.image2image_model}")
        resp = requests.post(model_api, json={"model_id": self.data.base_info.image2image_model})
        resp.raise_for_status()
        print(resp.json())

        def _generate_role_image(path_input_image, prompt, negative_prompt, path_output_image):
            """生成单个图像"""
            init_image = Image.open(path_input_image)
            image_b64 = self.pil_to_base64(init_image)
            prompt = ' '.join(prompt.split()[:max_length])
            negative_prompt = ' '.join(negative_prompt.split()[:max_length])
            print(f"正面提示词: {prompt}")
            print(f"负面提示词: {negative_prompt}")
            # 调用 FastAPI 生成图像
            image = self.image2image_via_api(
                prompt=prompt, 
                image_base64=image_b64,
                negative_prompt=negative_prompt,
                width=self.data.base_info.width,
                height=self.data.base_info.height,
                api_url=api_url,
                steps=30,
            )
            # display(image)
            # 转化image为统一的长宽
            resized_img = self.resize_keep_aspect(image=image, save_path=path_output_image)
            # display(resized_img)
            del image, resized_img
            gc.collect()
            torch.cuda.empty_cache()
        try:
            # if chatIDs is None:
            for scene, value in self.data.scripts.items():
                print(scene)
                shots = value['shots']
                for idx, chat_info in enumerate(shots):
                    if str(chat_info['type']) == "scenery":
                        chatID = chat_info['chatID']
                        path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}_scenery.png")
                        # 如果已经存在, 则跳过
                        if skip_if_exists and os.path.exists(path_output):
                            print(f"{chatID}_scenery已存在, 跳过生成。\n")
                            continue

                        path_input = os.path.join(self.data.base_info.OUTPUT_DIR, f"{scene}_opening_background.png")
                        # 判断是不是第一个节点
                        if idx == 0:
                            # 用开场背景图
                            shutil.copy(path_input, path_output)
                            results.append(path_output)
                        else:
                            prompt = f"High-definition landscape. {chat_info['prompt_en']}"
                            negative_prompt = f'''
Blurry, low quality, person, human, character, character shadows, figure, silhouette, man, woman, people, portrait, face, body, text, watermark, logo. {chat_info['negative_prompt_en']}
    '''
                            _generate_role_image(
                                path_input_image=path_input,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                path_output_image=path_output,
                            )
                            results.append(path_output)

    #         else:
    #             for chatID in chatIDs:
    #                 scene, _ = chatID.split('_')
    #                 shots = self.data.scripts[scene]['shots']
    #                 chat_info = [c for c in shots if (c['type']=='scenery' and c['chatID']==chatID)]
    #                 chat_info = chat_info[0] if chat_info else None
    #                 if chat_info is not None:
    #                     # 找到当前索引
    #                     idx = shots.index(chat_info)
    #                     path_input = os.path.join(self.data.base_info.OUTPUT_DIR, f"{scene}_opening_background.png")
    #                     path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}_scenery.png")
    #                     if idx == 0 or all(s['type'] != 'scenery' for s in shots[:idx]):
    #                         shutil.copy(path_input, path_output)
    #                         results.append(path_output)
    #                     else:
    #                         prompt = f"High-definition landscape. {chat_info['prompt_en']}"
    #                         negative_prompt = f'''
    # Blurry, low quality, person, human, character, character shadows, figure, silhouette, man, woman, people, portrait, face, body, text, watermark, logo. {chat_info['negative_prompt_en']}
    #     '''
    #                         _generate_role_image(
    #                             path_input_image=path_input,
    #                             prompt=prompt,
    #                             negative_prompt=negative_prompt,
    #                             path_output_image=path_output,
    #                         )
    #                         results.append(path_output)

        finally:
            print("请求 FastAPI 自我关闭, 释放显存...\n")
            # 等待所有请求完成
            while True:
                try:
                    resp = requests.get(f"http://{uvicorn_host}:{uvicorn_port}/busy", timeout=200)
                    if resp.status_code == 200 and not resp.json().get("busy", False):
                        break
                except Exception:
                    break
                print("FastAPI 正在推理, 等待完成再关闭...\n")
                time.sleep(20)

            try:
                requests.post(f"http://{uvicorn_host}:{uvicorn_port}/shutdown", timeout=5)
            except Exception as e:
                print("FastAPI 已经关闭或无法访问:", e)

            try:
                fastapi_process.wait(timeout=10)
                print("✅ FastAPI 已关闭")
            except Exception:
                print("FastAPI 未能在规定时间内退出, 执行强制终止...\n")
                fastapi_process.terminate()
                try:
                    fastapi_process.wait(timeout=5)
                except Exception:
                    print("❌ FastAPI 无法正常终止, 执行强制 kill")
                    fastapi_process.kill()
                print("FastAPI 已被强制关闭")
        return results
    # 5. 生成视频
    def generate_videos(self, skip_if_exists=True):
        print("[INFO] 正在生成视频...\n")
        self.data = self.data.load_json()
        results = []
        print_text = ''
        for text in ['opening', 'ending']:
            path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{text}_video.mp4")
            results.append(path_output)
        for scene, value in self.data.scripts.items():
            for role in value['shots']:
                chatID = role['chatID']
                save_file = os.path.abspath(os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}.mp4"))
                results.append(save_file)
        # 如果results中文件都已存在, 则直接跳过
        if all(os.path.exists(path) for path in results) and skip_if_exists:
            print("所有视频已存在, 跳过生成步骤。\n")
            return results, print_text
        # 生成片头片尾视频
        path_opening_scene_video =self.generate_opening_scene_video(skip_if_exists=skip_if_exists)
        # results = path_opening_scene_video
        # 生成对话视频
        # 生成multitalk的输入 JSON 文件
        for scene, value in self.data.scripts.items():
            print(scene)
            for role in value['shots']:
                if role['type'] == 'dialogue':
                    role_id = role['role_id']
                    chatID = role['chatID']
                    path_role = os.path.abspath(os.path.join(self.data.base_info.OUTPUT_DIR, f"{scene}_role_{role_id}.png"))
                    path_audio = os.path.abspath(os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}.wav"))
                    json_data = {
                        "prompt": DramaData.translate_text(role['表情和动作']),
                        "cond_image": path_role,
                        "cond_audio": {
                            "person1": path_audio
                        }
                    }
                    # 保存为 JSON 文件
                    json_path = os.path.abspath(os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}.json"))
                    with open(json_path, 'w') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=4)
                    print(f"已保存 JSON 文件: {json_path}")
                    
        path_chat_videos=self.generate_chat_videos(skip_if_exists=skip_if_exists)
        # results += path_chat_videos
        path_scenery_videos=self.generate_scene_scenery_video(skip_if_exists=skip_if_exists)
        # results += path_scenery_videos
        action_prompts = self.generate_action_prompts()

        for scene, value in self.data.scripts.items():
            for chat in value['shots']:
                if chat['type'] == "action":
                    # action 节点 prompt 使用 chatID
                    chat_id = chat['chatID']
                    chat['video_prompt'] = action_prompts[scene][chat_id]['正面提示词']
                    chat['video_negative_prompt'] = action_prompts[scene][chat_id]['负面提示词']
                    chat['video_prompt_en'] = DramaData.translate_text(chat['video_prompt'])
                    chat['video_negative_prompt_en'] = DramaData.translate_text(chat['video_negative_prompt'])
        path_action_videos=self.generate_scene_action_video(skip_if_exists=skip_if_exists)
        # results += path_action_videos
        for scene, value in self.data.scripts.items():
            print(scene)
            for chat in value['shots']:
                if chat['type'] == "action": 
                    chatID = chat['chatID']
                    path_action_video = os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}.mp4")
                    path_audio = os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}.wav")

                    if os.path.exists(path_action_video) and os.path.exists(path_audio):
                        # 输出临时文件
                        tmp_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}_tmp.mp4")

                        # ffmpeg 命令：把音频从 0 秒开始添加进视频
                        cmd = [
                            "ffmpeg", "-y",
                            "-i", path_action_video,
                            "-i", path_audio,
                            "-c:v", "copy",      # 保持视频编码不变
                            "-c:a", "aac",       # 音频转码为 aac, 保证兼容性
                            "-map", "0:v:0",     # 只取第一个输入的视频流
                            "-map", "1:a:0",     # 只取第二个输入的音频流
                            tmp_output
                        ]
                        try:
                            subprocess.run(cmd, check=True, capture_output=True, text=True)
                        except subprocess.CalledProcessError as e:
                            print("❌ ffmpeg 执行失败")
                            print("命令:", " ".join(cmd))
                            print("stderr:\n", e.stderr)
                            continue
                        # subprocess.run(cmd, check=True)
                        # 覆盖保存
                        os.replace(tmp_output, path_action_video)
                        print(f"已合成音频到视频：{path_action_video}")
                    else:
                        print(f"缺少文件: {path_action_video} 或 {path_audio}")
                        
        # 保存self.data到json文件
        self.data.save_json()
        print_text += f"✅ 视频生成完成\n"
        return results, print_text
    
    # 生成片头片尾视频
    def generate_opening_scene_video(self, skip_if_exists=True, max_length=500):
        ''' 
        :param skip_if_exists: 是否跳过已经存在的视频文件
        '''
        conda_env = "fast_video"
        uvicorn_host = "127.0.0.1"
        uvicorn_port = 5121
        api_url = f"http://{uvicorn_host}:{uvicorn_port}/image2video"
        model_api = f"http://{uvicorn_host}:{uvicorn_port}/load_model"
        results = []
        # ------------------- 启动 FastAPI -------------------
        fastapi_process = subprocess.Popen(
            [
                "conda", "run", "-n", conda_env,
                "uvicorn", "fastapi_image2video:app",
                "--host", uvicorn_host,
                "--port", str(uvicorn_port)
            ],
            # stdout=subprocess.PIPE, stderr=subprocess.PIPE
            stdout= None,
            stderr= None
        )
        print(f"FastAPI 已启动 (虚拟环境: {conda_env}), 等待模型加载...\n")

        # 等待服务就绪
        self.wait_for_fastapi_ready(api_url)

        # 加载指定模型
        print(f"请求 FastAPI 加载模型: {self.data.base_info.image2video_model}")
        resp = requests.post(model_api, json={"model_id": self.data.base_info.image2video_model})
        resp.raise_for_status()
        print(resp.json())

        try:
            for i in range(2):
                if i == 0:
                    text = 'opening'
                    path_input = os.path.join(self.data.base_info.OUTPUT_DIR, f"{text}_scene.png")
                else:
                    text = 'ending'
                    path_input = os.path.join(self.data.base_info.OUTPUT_DIR, f"{text}_background.png")
                path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{text}_video.mp4")
                # 如果文件已经存在, 则跳过
                if skip_if_exists and os.path.exists(path_output):
                    print(f"{text}_video.mp4已存在, 跳过生成。\n")
                    continue
                init_image = Image.open(path_input)
                image_b64 = self.pil_to_base64(init_image)
                prompt = f'''
        A short {text} scene video.
        Keep all characters in their original positions and maintain their appearance, clothing, and overall look.
        Allow natural and varied movements: blinking, subtle facial expressions, slight head tilts, gentle breathing, hand gestures, arm movements, torso shifts, small steps or turns, and natural posture adjustments.
        Background and environment remain mostly stable, with soft, gradual changes such as gentle light flicker, slow breeze, or subtle shadows.
        Focus on a cinematic {text} feel, with smooth and natural motion, cinematic lighting, and soft atmosphere.
                '''
                prompt = " ".join(prompt.split()[:max_length])
                negative_prompt = f'''
        Characters missing body parts, distorted limbs or faces.
        Drastic position changes between frames.
        Sudden scene/environment disappearance or major alterations.
        Characters moving unnaturally, excessively fast, or floating.
        Blurry or inconsistent features.
        Excessive camera shake, abrupt lighting changes, or jittery motion.
                '''
                negative_prompt = " ".join(negative_prompt.split()[:max_length])
                print('prompt:\n', prompt)
                print('negative_prompt:\n', negative_prompt)
                # 调用 FastAPI 生成图像
                output_video = self.image2video_via_api(
                    prompt=prompt, 
                    negative_prompt=negative_prompt,
                    image_base64=image_b64,
                    width=self.data.base_info.width,
                    height=self.data.base_info.height,
                    num_frames=81,
                    guidance_scale=5.0,
                    num_inference_steps=40,
                    api_url=api_url,
                    output_path=path_output
                )
                results.append(path_output)
                # 清理 GPU
                del init_image, image_b64
                gc.collect()
                torch.cuda.empty_cache()

        finally:
            print("请求 FastAPI 自我关闭, 释放显存...\n")
            # 等待所有请求完成
            while True:
                try:
                    resp = requests.get(f"http://{uvicorn_host}:{uvicorn_port}/busy", timeout=20)
                    if resp.status_code == 200 and not resp.json().get("busy", False):
                        break
                except Exception:
                    break
                print("FastAPI 正在推理, 等待完成再关闭...\n")
                time.sleep(20)

            try:
                requests.post(f"http://{uvicorn_host}:{uvicorn_port}/shutdown", timeout=5)
            except Exception as e:
                print("FastAPI 已经关闭或无法访问:", e)

            try:
                fastapi_process.wait(timeout=10)
                print("✅ FastAPI 已关闭")
            except Exception:
                print("FastAPI 未能在规定时间内退出, 执行强制终止...\n")
                fastapi_process.terminate()
                try:
                    fastapi_process.wait(timeout=5)
                except Exception:
                    print("❌ FastAPI 无法正常终止, 执行强制 kill")
                    fastapi_process.kill()
                print("FastAPI 已被强制关闭")
        return results
    @staticmethod
    def generate_multitalk_via_api(payload, api_url="http://127.0.0.1:5150/generate"):
        """调用 MultiTalk FastAPI 生成视频"""
        with requests.post(api_url, json=payload, stream=True) as resp:
            if resp.status_code != 200:
                print(f"[ERROR] 请求失败: {resp.status_code}, {resp.text}")
                return None

            print("[INFO] 开始流式接收日志 ...\n")
            try:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    print("[SERVER]", line)
            except Exception as e:
                print(f"[ERROR] 流式接收中断: {e}")

        # try:
        #     response = requests.post(api_url, json=payload)
        #     response.raise_for_status()
        #     result = response.json()
        #     cmd = result.get("cmd", None)
        #     print("FastAPI执行命令: ", " ".join(cmd))
        #     video_file = result.get("video_file", None)
        #     if video_file and video_file != "未返回文件":
        #         print("✅ 成功生成视频:", video_file)
        #     else:
        #         print("未生成文件:", payload.get("save_file", "unknown"))
        #         print(result)
        #     return video_file
        # except requests.HTTPError as e:
        #     print("❌ FastAPI 返回 HTTP 错误:", e)
        #     try:
        #         # 尝试打印详细响应内容
        #         print("返回内容:", response.text)
        #     except Exception:
        #         pass
        # except Exception as e:
        #     print("❌ 请求异常:", e)
        # return None

    def generate_chat_videos(self, skip_if_exists=True):
        """批量生成对话视频"""
        # ------------------- 启动 FastAPI -------------------
        conda_env = "multitalk"
        uvicorn_host = "127.0.0.1"
        uvicorn_port = 5122
        api_url = f"http://{uvicorn_host}:{uvicorn_port}/generate"
        results = []
        multi_talk_dir = os.path.abspath(os.path.join(self.data.base_info.BASE_DIR, "MultiTalk"))
        fastapi_process = subprocess.Popen(
            [
                "conda", "run", "-n", conda_env,
                "uvicorn", "fastapi_multitalk:app",
                "--host", uvicorn_host,
                "--port", str(uvicorn_port)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=multi_talk_dir  # 指定工作目录
        )
        print(f"FastAPI 已启动 (虚拟环境: {conda_env})")

        # 等待服务就绪
        self.wait_for_fastapi_ready(api_url)

        try:
            for scene, value in self.data.scripts.items():
                print(f"{scene}")
                for role in value['shots']:
                    if role['type'] == 'dialogue':
                        chatID = role['chatID']
                        path_output = os.path.abspath(os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}.mp4"))
                        # 如果文件已存在, 则跳过
                        if os.path.exists(path_output) and skip_if_exists:
                            print(f"{chatID}.mp4已存在, 跳过生成")
                            continue
                        json_path = os.path.abspath(os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}.json"))
                        save_file = os.path.abspath(os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}"))
                        ckpt_dir = os.path.abspath(os.path.join(self.data.base_info.BASE_DIR, "MultiTalk/weights/Wan2.1-I2V-14B-480P"))
                        wav2vec_dir = os.path.abspath(os.path.join(self.data.base_info.BASE_DIR, "MultiTalk/weights/chinese-wav2vec2-base"))
                        lora_dir = os.path.abspath(os.path.join(self.data.base_info.BASE_DIR, "MultiTalk/weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"))
                        # quant_dir = os.path.abspath(os.path.join(BASE_DIR, "MultiTalk/weights/MeiGen-MultiTalk"))

                        # payload 构建
                        payload = {
                            "ckpt_dir": ckpt_dir,
                            "wav2vec_dir": wav2vec_dir,
                            "input_json": json_path,
                            "lora_dir": lora_dir,
                            "lora_scale": 1.2,
                            "sample_text_guide_scale": 1.0,
                            "sample_audio_guide_scale": 2.0,
                            "sample_steps": 8,
                            "mode": "streaming",
                            "num_persistent_param_in_dit": 1,
                            "save_file": save_file,
                            "sample_shift": 2,
                            # "quant_dir": quant_dir,
                        }

                        # 调用 FastAPI 生成视频
                        print(f"生成视频: {save_file}.mp4")
                        self.generate_multitalk_via_api(payload, api_url)
                        results.append(f"{save_file}.mp4")
                        # 清理 GPU / 内存
                        gc.collect()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                        # 可选：短暂等待, 避免连续调用导致显存问题
                        time.sleep(3)
        
        finally:
            # ------------------- 关闭 FastAPI -------------------
            print("请求 FastAPI 关闭...\n")
            try:
                requests.post(f"http://{uvicorn_host}:{uvicorn_port}/shutdown", timeout=5)
            except Exception:
                pass

            try:
                fastapi_process.wait(timeout=10)
                print("✅ FastAPI 已关闭")
            except Exception:
                print("FastAPI 未能正常退出, 执行强制终止...\n")
                fastapi_process.terminate()
                try:
                    fastapi_process.wait(timeout=5)
                except Exception:
                    fastapi_process.kill()
                print("FastAPI 已被强制关闭")
        return results

    # 生成风景视频
    def generate_scene_scenery_video(self, skip_if_exists=True, max_length=500):
        ''' 
        :param scene_roles_id: 指定生成哪些角色, 比如 ["scene1_role_2", "scene2_role_3"]
        '''
        conda_env = "fast_video"
        uvicorn_host = "127.0.0.1"
        uvicorn_port = 5123
        api_url = f"http://{uvicorn_host}:{uvicorn_port}/image2video"
        model_api = f"http://{uvicorn_host}:{uvicorn_port}/load_model"
        results = []
        # ------------------- 启动 FastAPI -------------------
        fastapi_process = subprocess.Popen(
            [
                "conda", "run", "-n", conda_env,
                "uvicorn", "fastapi_image2video:app",
                "--host", uvicorn_host,
                "--port", str(uvicorn_port)
            ],
            # stdout=subprocess.PIPE, stderr=subprocess.PIPE
            stdout = None,
            stderr = None
        )
        print(f"FastAPI 已启动 (虚拟环境: {conda_env}), 等待模型加载...\n")

        # 等待服务就绪
        self.wait_for_fastapi_ready(api_url)

        # 加载指定模型
        print(f"请求 FastAPI 加载模型: {self.data.base_info.image2video_model}")
        resp = requests.post(model_api, json={"model_id": self.data.base_info.image2video_model})
        resp.raise_for_status()
        print(resp.json())

        try:
            for scene, value in self.data.scripts.items():
                print(scene)
                for chat_info in value['shots']:
                    if chat_info['type'] == "scenery":
                        chatID = chat_info['chatID']
                        path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}.mp4")
                        # 如果文件已存在, 跳过  
                        if skip_if_exists and os.path.exists(path_output):
                            print(f"{chatID}.mp4已存在, 跳过生成")
                            continue
                        # 读取图片
                        path_input = os.path.abspath(os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}_scenery.png"))
                        print(f"生成风景视频: {chatID}.mp4")
                        init_image = Image.open(path_input)
                        image_b64 = self.pil_to_base64(init_image)
                        prompt = chat_info['prompt_en']
                        prompt = f'''
    High-definition landscape. {prompt}
    '''
                        negative_prompt = chat_info['negative_prompt_en']
                        negative_prompt = " ".join(negative_prompt.split()[:max_length])
                        print('prompt:\n', prompt)
                        print('negative_prompt:\n', negative_prompt)
                        # 调用 FastAPI 生成视频
                        output_video = self.image2video_via_api(
                            prompt=prompt, 
                            negative_prompt=negative_prompt,
                            image_base64=image_b64,
                            width=self.data.base_info.width,
                            height=self.data.base_info.height,
                            num_frames=81,
                            guidance_scale=5.0,
                            num_inference_steps=40,
                            api_url=api_url,
                            output_path=path_output
                        )
                        results.append(path_output)
                        # 清理 GPU
                        del init_image
                        gc.collect()
                        torch.cuda.empty_cache()
        finally:
            print("请求 FastAPI 自我关闭, 释放显存...\n")
            # 等待所有请求完成
            while True:
                try:
                    resp = requests.get(f"http://{uvicorn_host}:{uvicorn_port}/busy", timeout=20)
                    if resp.status_code == 200 and not resp.json().get("busy", False):
                        break
                except Exception:
                    break
                print("FastAPI 正在推理, 等待完成再关闭...\n")
                time.sleep(20)

            try:
                requests.post(f"http://{uvicorn_host}:{uvicorn_port}/shutdown", timeout=5)
            except Exception as e:
                print("FastAPI 已经关闭或无法访问:", e)

            try:
                fastapi_process.wait(timeout=10)
                print("✅ FastAPI 已关闭")
            except Exception:
                print("FastAPI 未能在规定时间内退出, 执行强制终止...\n")
                fastapi_process.terminate()
                try:
                    fastapi_process.wait(timeout=5)
                except Exception:
                    print("❌ FastAPI 无法正常终止, 执行强制 kill")
                    fastapi_process.kill()
                print("FastAPI 已被强制关闭")
        return results
    
    # 生成动作视频
    # 生成每个场景中 action 节点的提示词, JSON 使用 chatID
    def generate_action_prompts(self):
        scripts = self.data.scripts
        # 提取所有风景节点的 chatID
        action_nodes = {
            scene: {chat["chatID"]: chat for chat in value["shots"] if chat["type"] == "action"}
            for scene, value in scripts.items()
        }
        messages = [
            {"role": "system", "content": f"你是一个视觉提示词专家, 擅长为AI绘画image2video生成正面和负面提示词, 用于生成真实自然的人物动作视频。"},
            {"role": "user", "content": 
            f'''
            剧本语言为{self.data.base_info.language}, 短剧风格为：{self.data.base_info.drama_style}, 文化背景为：{self.data.base_info.culture_background},
            剧本内容为：\n{action_nodes}\n,
            请为这些action节点生成正面提示词和负面提示词, 输出格式为：
            {{'scene1':{{chatID: {{'正面提示词': '...', '负面提示词': '...'}}}}, 'scene2':{{chatID: {{'正面提示词': '...', '负面提示词': '...'}}}}, ...}}
            其中 chatID 是节点的唯一标识符, 例如 scene1_0, scene2_1。
            正面提示词：详细描绘人物动作内容, 包括表情、身体姿势、动作细节、环境、光线氛围等, 要求画面清晰、真实自然, 符合{self.data.base_info.culture_background}背景和{self.data.base_info.drama_style}短剧风格；
            负面提示词：排除不需要的元素, 例如：模糊、畸形、低质量、身体缺失、违背文化背景、奇怪物体、背景复杂等。
            注意事项：
            - 提示词内容为{self.data.base_info.language},不得超过100个字；
            - 重点描写人物动作；
            - 要求详细直观描述, 不能使用缩写或成语。
            '''
            }
        ]
        result = self.generate_response(messages)
        return result

    def generate_scene_action_video(self, skip_if_exists=True, max_length=500):
        ''' 
        :param scene_roles_id: 指定生成哪些角色, 比如 ["scene1_role_2", "scene2_role_3"]
        '''
        conda_env = "fast_video"
        uvicorn_host = "127.0.0.1"
        uvicorn_port = 5124
        api_url = f"http://{uvicorn_host}:{uvicorn_port}/image2video"
        model_api = f"http://{uvicorn_host}:{uvicorn_port}/load_model"
        results = []
        # ------------------- 启动 FastAPI -------------------
        fastapi_process = subprocess.Popen(
            [
                "conda", "run", "-n", conda_env,
                "uvicorn", "fastapi_image2video:app",
                "--host", uvicorn_host,
                "--port", str(uvicorn_port)
            ],
            # stdout=subprocess.PIPE, stderr=subprocess.PIPE
            stdout= None,
            stderr= None
        )
        print(f"FastAPI 已启动 (虚拟环境: {conda_env}), 等待模型加载...\n")

        # 等待服务就绪
        self.wait_for_fastapi_ready(api_url)

        # 加载指定模型
        print(f"请求 FastAPI 加载模型: {self.data.base_info.image2video_model}")
        resp = requests.post(model_api, json={"model_id": self.data.base_info.image2video_model})
        resp.raise_for_status()
        print(resp.json())

        try:
            for scene, value in self.data.scripts.items():
                print(scene)
                for chat_info in value['shots']:
                    if chat_info['type'] != "action":
                        continue
                    chatID = chat_info['chatID']
                    role_id = chat_info['role_id']
                    path_output = os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}.mp4")
                    if skip_if_exists and os.path.exists(path_output):
                        print(f"{chatID}.mp4已存在, 跳过生成")
                        continue
                    path_input = os.path.abspath(
                        os.path.join(
                            self.data.base_info.OUTPUT_DIR, f"{scene}_role_{role_id}.png"
                            )
                        )
                    init_image = Image.open(path_input)
                    image_b64 = self.pil_to_base64(init_image)

                    prompt = chat_info['video_prompt_en']
                    prompt = f'''
    High-definition landscape, cinematic camera angle, dynamic composition.
    {prompt}'''
                    negative_prompt = chat_info['video_negative_prompt_en']
                    negative_prompt = " ".join(negative_prompt.split()[:max_length])
                    print('prompt:\n', prompt)
                    print('negative_prompt:\n', negative_prompt)
                    
                    # 调用 FastAPI 生成图像
                    output_video = self.image2video_via_api(
                        prompt=prompt, 
                        negative_prompt=negative_prompt,
                        image_base64=image_b64,
                        width=self.data.base_info.width,
                        height=self.data.base_info.height,
                        num_frames=81,
                        guidance_scale=5.0,
                        num_inference_steps=40,
                        api_url=api_url,
                        output_path=path_output
                    )
                    results.append(path_output)
                    # 清理 GPU
                    del init_image
                    gc.collect()
                    torch.cuda.empty_cache()
        finally:
            print("请求 FastAPI 自我关闭, 释放显存...\n")
            # 等待所有请求完成
            while True:
                try:
                    resp = requests.get(f"http://{uvicorn_host}:{uvicorn_port}/busy", timeout=20)
                    if resp.status_code == 200 and not resp.json().get("busy", False):
                        break
                except Exception:
                    break
                print("FastAPI 正在推理, 等待完成再关闭...\n")
                time.sleep(20)

            try:
                requests.post(f"http://{uvicorn_host}:{uvicorn_port}/shutdown", timeout=5)
            except Exception as e:
                print("FastAPI 已经关闭或无法访问:", e)

            try:
                fastapi_process.wait(timeout=10)
                print("✅ FastAPI 已关闭")
            except Exception:
                print("FastAPI 未能在规定时间内退出, 执行强制终止...\n")
                fastapi_process.terminate()
                try:
                    fastapi_process.wait(timeout=5)
                except Exception:
                    print("❌ FastAPI 无法正常终止, 执行强制 kill")
                    fastapi_process.kill()
                print("FastAPI 已被强制关闭")
        return results
    
    # 6. 拼接视频
    def concatenate_videos(self):
        print('[INFO] 正在拼接视频...')
        self.data = self.data.load_json()
        results = []
        print_text = ''
        # 添加标题文字到开场视频、结尾视频
        path_opening_video = os.path.join(self.data.base_info.OUTPUT_DIR, "opening_video.mp4")
        path_opening_video_with_title = os.path.join(self.data.base_info.OUTPUT_DIR, "opening_video_with_title.mp4")
        if os.path.exists(path_opening_video):
            self.add_title_to_video(
                path_input=path_opening_video,
                title_text=self.data.base_info['title'] + "\n\nFastVideo生成",
                font_file=self.get_fontfile(),
                path_output=path_opening_video_with_title
            )
        else:
            print(f"⚠️ 缺少开场视频: {path_opening_video}")

        path_ending_video = os.path.join(self.data.base_info.OUTPUT_DIR, "ending_video.mp4")
        path_ending_video_with_title = os.path.join(self.data.base_info.OUTPUT_DIR, "ending_video_with_title.mp4")
        if os.path.exists(path_ending_video):
            self.add_title_to_video(
                path_input=path_ending_video,
                title_text="剧终\n\nFastVideo生成",
                font_file=self.get_fontfile(),
                path_output=path_ending_video_with_title
            )
        else:
            print(f"⚠️ 缺少结尾视频: {path_ending_video}")

        all_clips = []
        TARGET_SIZE = (self.data.base_info.width, self.data.base_info.height)  # 宽, 高
        TARGET_FPS = 25

        # 开场视频
        final_opening_video = os.path.join(self.data.base_info.OUTPUT_DIR, "opening_video_complete.mp4")
        if os.path.exists(path_opening_video_with_title):
            self.resize_with_letterbox_ffmpeg(path_opening_video_with_title, final_opening_video, TARGET_SIZE, TARGET_FPS)
            all_clips.append(final_opening_video)
        else:
            pass

        # 场景视频
        for scene, value in self.data.scripts.items():
            print(f"处理场景: {scene}")
            for role in value.get('shots', []):
                chatID = role['chatID']
                subtitle = role.get('对话', None)
                path_chat_video = os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}.mp4")
                final_chat_video = os.path.join(self.data.base_info.OUTPUT_DIR, f"{chatID}_complete.mp4")
                if os.path.exists(path_chat_video):
                    self.resize_with_letterbox_ffmpeg(
                        path_chat_video, final_chat_video,
                        TARGET_SIZE, TARGET_FPS,
                        subtitle=subtitle, fontsize=36
                    )
                    all_clips.append(final_chat_video)
                else:
                    print(f"⚠️ 缺少场景视频: {path_chat_video}, 已跳过。")

        # 结尾视频
        final_ending_video = os.path.join(self.data.base_info.OUTPUT_DIR, "ending_video_complete.mp4")
        if os.path.exists(path_ending_video_with_title):
            self.resize_with_letterbox_ffmpeg(path_ending_video_with_title, final_ending_video, TARGET_SIZE, TARGET_FPS)
            all_clips.append(final_ending_video)
        else:
            pass

        # 拼接
        if all_clips:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = os.path.join(self.data.base_info.OUTPUT_DIR, f"FastVideo_{timestamp}.mp4")
            self.concat_videos_ffmpeg(all_clips, output_path)
            print(f"✅ 所有视频已拼接完成！输出文件: {output_path}")
            self.output_video_path = output_path
            results.append(output_path)
            print_text = f"✅ 所有视频已拼接完成！输出文件: {output_path}"
        else:
            print("❌ 未找到可拼接的视频文件！")
            print_text = "❌ 未找到可拼接的视频文件！"
        return results, print_text
                    
    @staticmethod
    def sharpness_score(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 将方差映射到 0-100（启发式, 阈值可调）
        minv, maxv = 10.0, 400.0
        s = (var - minv) / (maxv - minv) * 100.0
        return float(np.clip(s, 0, 100))
    @staticmethod
    def contrast_score(image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        std = float(np.std(gray))
        minv, maxv = 5.0, 100
        s = (std - minv) / (maxv - minv) * 100.0
        return float(np.clip(s, 0, 100))

    # -------- CLIP 相似度（0-100） --------
    def clip_similarity_score(self, image_path, prompt="a beautiful character portrait"):
        model, preprocess = clip.load("ViT-B/32", device=self.data.base_info.device)
        img = Image.open(image_path).convert("RGB")
        image = preprocess(img).unsqueeze(0).to(self.data.base_info.device)
        for n in range(77, 0, -1):
            try:
                prompt_trunc = ' '.join(prompt.split()[:n])
                text = clip.tokenize([prompt_trunc]).to(self.data.base_info.device)
                break
            except RuntimeError:
                continue
        with torch.no_grad():
            img_feat = model.encode_image(image)
            txt_feat = model.encode_text(text)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).item()  # -1..1
        # 释放显存
        del model, preprocess
        gc.collect()
        torch.cuda.empty_cache()
        return float((sim) * 200.0)

    # -------- YOLO 人物检测评分 --------
    @staticmethod
    def yolo_person_score(image_path):
        yolo_model = YOLO("yolov8n-seg.pt")
        results = yolo_model(image_path, verbose=False)[0]
        if results.masks is None:
            return 0.0  # 没有人物
        img_h, img_w = results.orig_shape
        img_area = img_h * img_w
        best_score = 0.0
        for mask, conf, cls in zip(results.masks.data, results.boxes.conf, results.boxes.cls):
            if int(cls.item()) != 0:  # 0 = person
                continue
            mask_area = mask.sum().item()
            ratio = mask_area / img_area  # 人物像素占比
            # 要求人物占比适中 0.2
            ratio_score = 100 * (1 - abs(ratio - 0.1) / 0.1)  # 惩罚过大过小
            ratio_score = np.clip(ratio_score, 0, 100)
            score = float(conf.item() * 100 * 0.6 + ratio_score * 0.4)
            best_score = max(best_score, score)
        del yolo_model
        gc.collect()
        torch.cuda.empty_cache()
        return best_score

    # -------- 最终评分器（可自定义权重） --------
    def score_character(self,
                        image_path,
                        prompt="a beautiful character portrait",
                        weights=None):
        """
        weights: dict
        """
        if weights is None:
            weights = {"clip":0.15, "sharp":0.25, "contrast":0.2, "yolo":0.4}

        clip_s = self.clip_similarity_score(image_path, prompt)
        sharp_s = self.sharpness_score(image_path)
        contrast_s = self.contrast_score(image_path)
        yolo_s = self.yolo_person_score(image_path)

        total_w = sum(weights.values())

        if total_w <= 0:
            raise ValueError("总权重为 0, 请检查 weights 参数。")

        s = (
            clip_s * weights.get("clip", 0) +
            sharp_s * weights.get("sharp", 0) +
            contrast_s * weights.get("contrast", 0) +
            yolo_s * weights.get("yolo", 0)
        )

        final = s / total_w
        components = {
            "clip": round(clip_s, 2),
            "sharp": round(sharp_s, 2),
            "contrast": round(contrast_s, 2),
            "yolo": round(yolo_s, 2)
        }
        return round(final,2), components

    @staticmethod
    def extract_person(input_path, output_path):
        """
        从 PNG 图片中扣取人物并保存为透明背景的 PNG
        :param input_path: 输入 PNG 文件路径
        :param output_path: 输出 PNG 文件路径
        """
        # 读取原图
        with open(input_path, 'rb') as inp_file:
            input_data = inp_file.read()

        # 扣图
        result_data = remove(
            input_data,
            alpha_matting=True,                     # 开启 Matting
            alpha_matting_foreground_threshold=50, # 默认 240, 调低更宽松
            # alpha_matting_background_threshold=10,  # 默认 10, 调高保留更多
            # alpha_matting_erode_size=1,             # 默认 10, 调小减少过度腐蚀
            # post_process_mask=True                 # 关闭后处理, 避免裁掉细节
            )  # rembg 会自动处理并输出透明背景的 RGBA 图

        # 转成 PIL 图片
        result_image = Image.open(io.BytesIO(result_data)).convert("RGBA")
        
        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_image.save(output_path, "PNG")
        print(f"扣图完成, 保存到: {output_path}")
        # display(result_image)
        return result_image

    @staticmethod
    def resize_keep_aspect(image: Image.Image, save_path: str, target_size=(720, 1280), mode="scale", fill_color=(0,0,0)):
        """
        将图片转换为指定分辨率的三种模式:
        
        mode="scale" : 按比例缩放, 外切目标尺寸, 裁掉多余部分 (推荐商业应用)
        mode="pad"   : 按比例缩放, 内切目标尺寸, 补边到目标尺寸
        mode="resize": 直接缩放到目标尺寸（不保持比例, 可能变形）
        
        :param image: 输入 PIL Image
        :param save_path: 输出路径
        :param target_size: 目标尺寸 (width, height)
        :param mode: 转换模式 ["scale", "pad", "resize"]
        :param fill_color: 补边颜色, 默认黑色
        :return: 转换后的 PIL Image
        """
        target_w, target_h = target_size
        # w, h = image.size
        
        if mode == "scale":
            # 等比缩放并裁剪
            image = ImageOps.fit(image, target_size, Image.LANCZOS, centering=(0.5, 0.5))
            print(f"[scale] 缩放并裁剪为 {target_w}x{target_h}")
        
        elif mode == "pad":
            # 等比缩放并补边
            image.thumbnail(target_size, Image.LANCZOS)
            # 根据输入图像模式选择新画布模式
            if image.mode == "RGBA":
                new_img = Image.new("RGBA", target_size, (0,0,0,0))  # 保留透明
            else:
                new_img = Image.new("RGB", target_size, fill_color)

            paste_x = (target_w - image.width) // 2
            paste_y = (target_h - image.height) // 2
            new_img.paste(image, (paste_x, paste_y))
            image = new_img
            print(f"[pad] 缩放并补边为 {target_w}x{target_h}")
        
        elif mode == "resize":
            # 直接缩放, 不保持比例
            image = image.resize(target_size, Image.LANCZOS)
            print(f"[resize] 强制缩放为 {target_w}x{target_h}")
        
        else:
            raise ValueError("mode 必须是 'scale', 'pad', 或 'resize'")
        
        image.save(save_path)
        return image
    @staticmethod
    def pil_to_base64(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    @staticmethod
    def wait_for_fastapi_ready(api_url, timeout=600, interval=10):
        """
        轮询检测 FastAPI 服务是否就绪
        :param api_url: FastAPI API URL, 可能以 /generate, /text2image, /image2image 等结尾
        """
        start = time.time()
        # 正则替换最后的/* 为 /health
        health_url = re.sub(r'/[^/]+$', '/health', api_url)
        print(health_url)
        while time.time() - start < timeout:
            try:
                # health_url = api_url.replace("/generate", "/health")  # 健康检查接口
                resp = requests.get(health_url, timeout=2)

                if resp and resp.status_code == 200:
                    print("✅ FastAPI 已就绪, 可以接收请求")
                    return True
            except Exception:
                pass

            print("⏳ 等待 FastAPI 就绪...\n")
            time.sleep(interval)

        raise TimeoutError("FastAPI 服务启动超时")
    @staticmethod
    def text2image_via_api(prompt, negative_prompt="", width=512, height=512, steps=20, api_url="http://127.0.0.1:5101/text2image"):
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps
        }
        resp = requests.post(api_url, json=payload)
        resp.raise_for_status()
        resp_json = resp.json()
        if "image_base64" not in resp_json:
            raise ValueError(resp_json.get("error", "Unknown error"))
        
        img_data = base64.b64decode(resp_json["image_base64"])
        image = Image.open(io.BytesIO(img_data))
        return image
    @staticmethod
    def image2image_via_api(prompt, image_base64, negative_prompt="", width=512, height=512, steps=20, api_url="http://127.0.0.1:5111/image2image"):
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_base64": image_base64,
            "width": width,
            "height": height,
            "num_inference_steps": steps
        }
        resp = requests.post(api_url, json=payload)
        resp.raise_for_status()
        resp_json = resp.json()
        if "image_base64" not in resp_json:
            raise ValueError(resp_json.get("error", "Unknown error"))
        
        img_data = base64.b64decode(resp_json["image_base64"])
        image = Image.open(io.BytesIO(img_data))
        return image
    @staticmethod
    def image2video_via_api(
            prompt, image_base64, output_path, 
            negative_prompt="", width=512, height=512, 
            num_frames=81, guidance_scale=5.0, num_inference_steps=50, 
            api_url="http://127.0.0.1:5121/image2video"
            ):
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_base64": image_base64,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "output_path": output_path,
        }
        resp = requests.post(api_url, json=payload)
        resp.raise_for_status()
        resp_json = resp.json()
        if "video_path" not in resp_json:
            raise ValueError(resp_json.get("error", "Unknown error"))
        
        return resp_json["video_path"]
    @staticmethod
    def remove_persons(
        input_path, 
        output_path, 
        dilate_iter=5, 
        blur_size=7, 
        conf_thres=0.2, 
        mode="union"  # 新增参数, 默认 intersection
    ):
        """
        根据 mode 选择不同的去人物方式:
        - rembg: 只用 rembg 抠图结果
        - yolo: 只用 YOLOv8-seg 检测人物
        - intersection: rembg ∩ yolo （严格）
        - union: rembg ∪ yolo （宽松）
        """
        # -------- Step 1: rembg 初步抠图 --------
        with open(input_path, "rb") as inp_file:
            input_data = inp_file.read()

        result_data = remove(
            input_data,
            alpha_matting=True,
            alpha_matting_foreground_threshold=150,
        )
        rembg_rgba = Image.open(io.BytesIO(result_data)).convert("RGBA")
        rembg_np = np.array(rembg_rgba)

        h, w = rembg_np.shape[:2]
        alpha = rembg_np[:, :, 3]

        # rembg 的前景 mask
        mask_rembg = np.where(alpha > 0, 255, 0).astype(np.uint8)

        # -------- Step 2: YOLOv8n-seg 检测人物 --------
        yolo_model = YOLO("yolov8n-seg.pt")
        results = yolo_model.predict(input_path, conf=conf_thres, verbose=False, iou=0.5)
        # annotated_img = Image.fromarray(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
        # display(annotated_img)

        mask_yolo = np.zeros((h, w), dtype=np.uint8)
        for r in results:
            if r.masks is None:
                continue
            for mask, cls in zip(r.masks.data, r.boxes.cls):
                if int(cls) == 0:  # class=0 是 person
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                    mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_yolo = cv2.bitwise_or(mask_yolo, mask_resized)

        # -------- Step 3: 根据 mode 选择 mask --------
        if mode == "rembg":
            final_mask = mask_rembg
        elif mode == "yolo":
            final_mask = mask_yolo
        elif mode == "intersection":
            final_mask = cv2.bitwise_and(mask_rembg, mask_yolo)
        elif mode == "union":
            final_mask = cv2.bitwise_or(mask_rembg, mask_yolo)
        else:
            raise ValueError(f"Invalid mode: {mode}, must be one of ['rembg', 'yolo', 'intersection', 'union']")

        # 膨胀 + 模糊
        kernel = np.ones((7, 7), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        final_mask = cv2.dilate(final_mask, kernel, iterations=dilate_iter)
        final_mask = cv2.GaussianBlur(final_mask, (blur_size, blur_size), 0)

        # -------- Step 4: 应用 mask --------
        bg = Image.open(input_path).convert("RGBA")
        bg_np = np.array(bg)
        bg_np[final_mask > 128] = (0, 0, 0, 0)

        # -------- Step 5: 保存 --------
        result = Image.fromarray(bg_np)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.save(output_path, "PNG")

        print(f"人物已去除（mode={mode}）, 保存到: {output_path}")
        del yolo_model
        gc.collect()
        torch.cuda.empty_cache()
        return result
    @staticmethod
    def get_fontfile():
        """
        自动获取系统可用的中文字体路径
        优先使用 ~/.fonts/SimHei.ttf
        """
        # 优先选择用户自带字体
        simhei_path = os.path.expanduser("~/.fonts/SimHei.ttf")
        if os.path.exists(simhei_path):
            return simhei_path

        candidates = []

        # 使用 fc-list 搜索系统中文字体
        try:
            out = subprocess.check_output(
                ["fc-list", ":lang=zh", "-f", "%{file}\\n"],
                stderr=subprocess.DEVNULL
            ).decode("utf-8", errors="ignore").splitlines()

            seen = set()
            for p in out:
                if p and p not in seen and os.path.exists(p):
                    candidates.append(p)
                    seen.add(p)
        except Exception:
            pass

        # 常见 fallback 路径
        fallbacks = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
        for p in fallbacks:
            if p not in candidates and os.path.exists(p):
                candidates.append(p)

        return candidates[0] if candidates else None
    @staticmethod
    def add_title_to_video(path_input, title_text, font_file, path_output):
        """
        给视频添加中文标题
        """
        # ffmpeg 命令
        cmd = [
            "ffmpeg", "-y",
            "-i", path_input,
            "-vf",
            f"drawtext=fontfile='{font_file}':text='{title_text}':x=(w-text_w)/2:y=h*0.2:fontsize=48:fontcolor=white:shadowx=2:shadowy=2",
            "-codec:a", "copy",
            path_output
        ]

        # print("运行命令:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("生成完成:", path_output)
    @staticmethod
    def auto_wrap_subtitle(subtitle, target_w, fontsize, margin_ratio=0.9):
        """根据屏幕宽度和字体大小自动换行字幕"""
        avg_char_width = fontsize * 1.0
        max_width = target_w * margin_ratio
        max_chars_per_line = max(1, int(max_width // avg_char_width))
        wrapped = "\n".join(textwrap.wrap(subtitle, width=max_chars_per_line))
        return wrapped
    @staticmethod
    def _has_audio_stream(path):
        p = subprocess.run(
            ["ffprobe", "-i", path, "-show_streams", "-select_streams", "a", "-loglevel", "error"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return bool(p.stdout.strip())

    def resize_with_letterbox_ffmpeg(self, input_path, output_path, target_size, target_fps, subtitle=None, fontsize=32):
        """等比例缩放 + 黑边填充到目标分辨率, 并保留或添加音频；返回 True/False"""
        target_w, target_h = target_size
        vf_filters = [
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease",
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
        ]

        # 处理字幕：自动换行并对换行/冒号等进行转义
        if subtitle:
            subtitle = subtitle.replace("'", "’")  # 替换单引号避免冲突
            wrapped = self.auto_wrap_subtitle(subtitle, target_w, fontsize)
            fontfile = self.get_fontfile()
            if fontfile:
                font_arg = f":fontfile={fontfile}"
            else:
                # If nothing found, skip drawtext to avoid ffmpeg failure
                print("Warning: no usable font found for drawtext; skipping subtitles. Install fonts (e.g. fonts-dejavu) or set fontfile.")
                font_arg = None

            if font_arg:
                if "\n" in wrapped:
                    tf = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt")
                    tf.write(wrapped)
                    tf.flush()
                    tf.close()
                    tf_path = tf.name
                    vf_filters.append(
                        f"drawtext=textfile={tf_path}:reload=1:fontfile={fontfile}:fontcolor=white:fontsize={fontsize}:"
                        f"borderw=2:x=(w-text_w)/2:y=h-140:line_spacing=10:box=1:boxcolor=black@0.4:boxborderw=5"
                    )
                else:
                    wrapped_escaped = wrapped.replace("\\", "\\\\").replace("\n", "\\n").replace(":", "\\:")
                    vf_filters.append(
                        f"drawtext=text='{wrapped_escaped}':fontfile={fontfile}:fontcolor=white:fontsize={fontsize}:"
                        f"borderw=2:x=(w-text_w)/2:y=h-140:line_spacing=10:box=1:boxcolor=black@0.4:boxborderw=5"
                    )

        # 检查输入是否有音频
        has_audio = self._has_audio_stream(input_path)

        # ...existing code...
        if has_audio:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-vf", ",".join(vf_filters),
                "-r", str(target_fps),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "192k",
                "-map", "0:v", "-map", "0:a:0",
                "-shortest",
                output_path
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-vf", ",".join(vf_filters),
                "-r", str(target_fps),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "192k",
                "-map", "0:v", "-map", "1:a",
                "-shortest",
                output_path
            ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"ffmpeg failed for {input_path} -> {output_path}")
            print(res.stderr.decode(errors="ignore"))
            return False
        return True
    
    @staticmethod
    def concat_videos_ffmpeg(video_list, output_path):
        """用 ffmpeg 拼接多个视频, 统一转码以提高兼容性"""
        # 创建临时 filelist
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as f:
            for v in video_list:
                f.write(f"file '{os.path.abspath(v)}'\n")
            list_file = f.name

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", list_file,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print("ffmpeg concat failed:")
            print(res.stderr.decode(errors="ignore"))
            return False
        # 可选：os.remove(list_file)
        return True

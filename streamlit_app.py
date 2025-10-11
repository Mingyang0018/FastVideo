import streamlit as st
from main import DramaData, BaseInfo, DramaPipeline, StreamlitLogger
import os
from PIL import Image
import time
import sys
from collections import defaultdict
st.set_page_config(page_title="FastVideo", layout="wide")

# ---------------------------
# 左侧边栏：配置参数（只是收集输入，还没实例化 BaseInfo）
# ---------------------------
dict_zh_en = {
    "中国古代": "Chinese ancient",
    "中国现代": "Chinese modern",
    "西方古代": "Western ancient",
    "西方现代": "Western modern",
    "动画效果": "Animated movie",
    "真人效果": "Live-action movie",
}
with st.sidebar:
    st.header("配置参数")
    model_vendor = st.selectbox(
        "大模型厂商",
        options=["DeepSeek"],
        index=0,
        help="选择用于生成剧本和提示词的大语言模型"
    )
    api_key = st.text_input("API Key", value='', type="password")
    # 根据model_vendor选择model_id，DeepSeek: [deepseek-chat, deepseek-reasoner]
    dict_model_id = {
        'DeepSeek': ['deepseek-chat','deepseek-reasoner']
    }
    model_id = 'deepseek-chat'
    if model_vendor:
        model_id = st.selectbox(
            "模型", dict_model_id[model_vendor]
        )
    language = st.selectbox("语言", ["中文", "English"])
    culture_background_zh = st.selectbox("文化背景", ["中国古代","中国现代","西方古代","西方现代"], index=0)
    culture_background = dict_zh_en.get(culture_background_zh, "Chinese ancient")
    drama_style_zh = st.selectbox("短剧风格", ["动画效果", "真人效果"], index=0)
    drama_style = dict_zh_en.get(drama_style_zh, "Animated movie")
    # --- 新增：关键词多选 ---
    keywords = st.multiselect(
        "关键词",
        options=["喜剧", "悬疑", "科幻", "爱情", "动作", "奇幻", "剧情", "励志", "恐怖"],
        default=["喜剧"],
        help="可选择多个关键词来指定短剧风格方向"
    )

    resolution = st.selectbox("分辨率", [(720, 1280), (1280, 720), (1080, 1080), (512, 512)], index=0)
    width, height = resolution
    # 计算角色尺寸，保持 9:16的宽高比，缩放为resolution的宽或高，不得超出resolution
    role_width = min(width, height / 16 * 9)
    role_height = role_width / 9 * 16
    role_size = (int(role_width), int(role_height))

    text2image_model = st.selectbox("text-to-image 模型", ["black-forest-labs/FLUX.1-dev"], index=0)
    text2image_model = "./" + text2image_model.split("/")[-1]

    image2image_model = st.selectbox("image-to-image 模型", ["black-forest-labs/FLUX.1-Kontext-dev"], index=0)
    image2image_model = "./" + image2image_model.split("/")[-1]

    image2video_model = st.selectbox("image-to-video 模型", ["Wan-AI/Wan2.2-TI2V-5B-Diffusers"], index=0)
    image2video_model = "./" + image2video_model.split("/")[-1]
    # 添加代理设置部分
    proxy_enabled = st.checkbox("开启代理", value=False)  # 布尔值控件，用于开启或关闭代理
    proxy_port = None
    if proxy_enabled:
        proxy_port = st.number_input("代理端口", min_value=1, max_value=65535, value=7897)  # 代理端口输入框

# ---------------------------
# 中间区域：主题和时长
st.title("FastVideo 短剧生成")
st.markdown("配置参数 → 输入主题与时长 → 一键生成自动化生成短剧 → 分步生成逐步生成短剧")

col1, col2 = st.columns([5,1])
with col1:
    theme = st.text_input("主题", placeholder="请输入短剧主题")
with col2:
    time_limit = st.number_input("时长 (分钟)", min_value=1, max_value=60, value=5)

one_click = st.button("一键生成", type="primary")

# 日志展示
def show_log():
    log_placeholder = st.empty()
    # 初始化 logger
    logger = StreamlitLogger(log_placeholder)
    sys.stdout = logger   # 把所有 print 输出到前端
    sys.stderr = logger  # 把所有错误输出到前端
# ---------------------------
# 初始化检查 + 会话状态保存
def check_and_init(force_refresh=True, load_data=True, allow_partial_load=False):
    """
    force_refresh: 是否强制重新初始化 pipeline
    load_data: 是否加载已有 drama_data.json
    allow_partial_load: 是否允许在未填写完整参数时，仅加载已有文件
    """
    if not api_key or not theme:
        if allow_partial_load:
            # 只加载已有 drama_data.json
            if "drama_data" in st.session_state:
                drama_data = st.session_state["drama_data"]
            else:
                # 尝试从 base_info.path_drama_data 加载
                if "base_info" in st.session_state and os.path.exists(st.session_state["base_info"].path_drama_data):
                    drama_data = DramaData(base_info=st.session_state["base_info"]).load_json()
                    st.session_state["drama_data"] = drama_data
                else:
                    drama_data = None
            return None, drama_data, None
        else:
            st.warning("请填写完整配置参数（API Key、主题）")
            return None, None, None

    # 如果已经存在则直接返回
    if "drama_pipeline" in st.session_state and not force_refresh:
        return (
            st.session_state["base_info"],
            st.session_state["drama_data"],
            st.session_state["drama_pipeline"],
        )
    # 初始化
    base_info = BaseInfo(
        api_key=api_key,
        model_id=model_id,
        language=language,
        culture_background=culture_background,
        drama_style=drama_style,
        keywords=keywords,
        width=width,
        height=height,
        text2image_model=text2image_model,
        image2image_model=image2image_model,
        image2video_model=image2video_model,
        theme=theme,
        time_limit=time_limit,
        use_proxy=proxy_enabled,
        proxy_port=proxy_port
    )
    drama_data = DramaData(base_info=base_info)
    # 如果存在数据base_info.path_drama_data,并且load_data为True，则加载
    if load_data and os.path.exists(base_info.path_drama_data):
        drama_data = drama_data.load_json()
        # drama_data.base_info = base_info  # 更新 base_info
    drama_pipeline = DramaPipeline(data=drama_data)
    # 保存到 session_state
    st.session_state["base_info"] = base_info
    st.session_state["drama_data"] = drama_data
    st.session_state["drama_pipeline"] = drama_pipeline
    return base_info, drama_data, drama_pipeline

# 定义 results，集中管理每个步骤的文件路径
def get_results():
    base_info, drama_data, _=check_and_init(force_refresh=True, load_data=True, allow_partial_load=True)
    # st.write(drama_data.roles)
    if drama_data is None:
        return {
            "script": [],
            "voices": [],
            "roles": [],
            "scenes": [],
            "videos": []
        }
    results = {
        "script": [
            drama_data.base_info.path_script,
            drama_data.base_info.path_drama_data,
        ],
        "voices": [],
        "roles": [],
        "scenes": [],
        "videos": []
    }
    # voices
    if drama_data.scripts and drama_data.roles:
        for role_id, role in drama_data.roles.items():
            path_voice = os.path.join(drama_data.base_info.OUTPUT_DIR, f'voice_{role_id}.wav')
            results['voices'].append(path_voice)
        for key, value in drama_data.scripts.items():
            for index, plot in enumerate(value['shots']):
                chatID = plot['chatID']
                if plot["type"] != "scenery":
                    path_audio = os.path.join(drama_data.base_info.OUTPUT_DIR, f'{chatID}.wav')
                    results['voices'].append(path_audio)
    # roles
    if drama_data.roles:
        for role_id, role in drama_data.roles.items():
            path_role = os.path.join(drama_data.base_info.OUTPUT_DIR, f'{role_id}.png')
            name = role.get('name', None)
            results['roles'].append(
                {
                    'role_id': role_id,
                    'name': name,
                    'path': path_role,
                    'exists': os.path.exists(path_role)
                }
                                    )
    # scenes
    for text in ['opening', 'ending']:
        path_output = os.path.join(drama_data.base_info.OUTPUT_DIR, f"{text}_background.png")
        results['scenes'].append(path_output)
    results['scenes'] += [(os.path.join(drama_data.base_info.OUTPUT_DIR, "opening_scene.png"))]
    for scene, value in drama_data.scripts.items():
            path_output = os.path.join(drama_data.base_info.OUTPUT_DIR, f"{scene}_opening_background.png")
            results['scenes'].append(path_output)
    for scene, value in drama_data.scripts.items():
        for scene_role in value['scene_roles']:
            role_id = scene_role['role_id']
            final_path = os.path.join(drama_data.base_info.OUTPUT_DIR, f"{scene}_role_{role_id}.png")
            results['scenes'].append(final_path)
    for scene, value in drama_data.scripts.items():
        for idx, chat_info in enumerate(value['shots']):
            if str(chat_info['type']) == "scenery":
                chatID = chat_info['chatID']
                path_output = os.path.join(drama_data.base_info.OUTPUT_DIR, f"{chatID}_scenery.png")
                results['scenes'].append(path_output)
    # videos
    for text in ['opening', 'ending']:
        path_output = os.path.join(drama_data.base_info.OUTPUT_DIR, f"{text}_video.mp4")
        results['videos'].append(path_output)
    for scene, value in drama_data.scripts.items():
        for role in value['shots']:
            chatID = role['chatID']
            save_file = os.path.abspath(os.path.join(drama_data.base_info.OUTPUT_DIR, f"{chatID}.mp4"))
            results['videos'].append(save_file)
    return results
path_output = get_results()
# ---------------------------
# 通用删除确认函数
def confirm_delete(item_name, session_state_key=None, extra_remove_keys=None):
    """
    item_name: str，删除对象名称，如 "剧本"、"角色"
    session_state_key: str，可选，删除后从 session_state 删除的 key
    extra_remove_keys: list，可选，删除后从 session_state 删除的额外 key 列表
    """
    if st.session_state.get("confirm_action") != f"delete_{item_name}":
        return
    file_list = path_output[item_name]
    if item_name == "roles":
        file_list=path_output['roles'] + path_output['voices']
    placeholder = st.empty()
    with placeholder.container():
        st.warning(f"是否删除全部{item_name}？")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 是", key=f"confirm_del_{item_name}"):
                st.session_state["confirm_action"] = f"confirm_del_{item_name}"
                
        with col2:
            if st.button("❌ 否", key=f"cancel_del_{item_name}"):
                st.session_state["confirm_action"] = None
                placeholder.empty()

    if st.session_state["confirm_action"] == f"confirm_del_{item_name}":
        placeholder.empty()
        deleted = []
        for f in file_list:
            if isinstance(f, dict):  # 如果传入的是 dict 列表（如角色）
                f_path = f.get('path')
                if f_path and os.path.exists(f_path):
                    os.remove(f_path)
                    deleted.append(f_path)
            elif os.path.exists(f):
                os.remove(f)
                deleted.append(f)
        if session_state_key:
            st.session_state.pop(session_state_key, None)
        if extra_remove_keys:
            for k in extra_remove_keys:
                st.session_state.pop(k, None)
        st.session_state["confirm_action"] = None
        if deleted:
            st.success(f"✅ 已删除 {len(deleted)} 个 {item_name} 文件")
        else:
            st.warning(f"没有找到可删除的 {item_name} 文件")
        time.sleep(1)
        st.rerun()
# ---------------------------
# 删除工具函数
def delete_files(file_list):
    deleted = []
    for path_file in file_list:
        if os.path.exists(path_file):
            os.remove(path_file)
            deleted.append(path_file)
    return deleted

# 初始化 session_state
if "confirm_action" not in st.session_state:
    st.session_state["confirm_action"] = None

# 页面最开始初始化 pipeline（只初始化一次，保存在 session_state）
base_info, drama_data, drama_pipeline = check_and_init(force_refresh=False, load_data=True, allow_partial_load=True)

# 一键生成短剧
def generate_drama_all(drama_pipeline, skip_if_exists=False):
    show_log()
    drama_pipeline.generate_script_and_shot(skip_if_exists=skip_if_exists)
    drama_pipeline.generate_voices_sync(skip_if_exists=skip_if_exists)
    drama_pipeline.generate_roles(skip_if_exists=skip_if_exists)
    drama_pipeline.generate_scenes(skip_if_exists=skip_if_exists)
    drama_pipeline.generate_videos(skip_if_exists=skip_if_exists)
    drama_pipeline.concatenate_videos()
    if drama_pipeline.output_video_path:
        st.subheader("最终结果")
        st.video(drama_pipeline.output_video_path)
        st.download_button(
            "下载视频", 
            data=b"", 
            file_name=os.path.basename(drama_pipeline.output_video_path)
            )
    st.session_state["confirm_action"] = None
    time.sleep(1)
    st.rerun()
# ---------------------------
# 一键生成
if one_click:
    if not api_key or not theme:
        st.warning("请填写完整配置参数（API Key、主题）")
    else:
        st.session_state["confirm_action"] = "generate_all"
if st.session_state["confirm_action"] == "generate_all":
    placeholder = st.empty()
    with placeholder.container():
        st.warning("请选择生成模式：")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            if st.button("✅ 覆盖生成(从头开始重新生成)", key="confirm_generate_all_overwrite"):
                st.session_state["confirm_action"] = "confirm_generate_all_overwrite"
        with col_c2:
            if st.button("✅ 补充生成(跳过已生成的部分)", key="confirm_generate_all_skip"):
                st.session_state["confirm_action"] = "confirm_generate_all_skip"
        with col_c3:
            if st.button("❌ 取消", key="cancel_generate_all"):
                st.session_state["confirm_action"] = None
                placeholder.empty()
if st.session_state["confirm_action"] == "confirm_generate_all_overwrite":
    st.session_state["confirm_action"] = None
    placeholder.empty()
    skip_if_exists = False
    st.success("正在覆盖生成视频.................................................")
    base_info, drama_data, drama_pipeline = check_and_init(force_refresh=True, load_data=False)
    if drama_pipeline is not None:
        generate_drama_all(drama_pipeline=drama_pipeline, skip_if_exists=skip_if_exists)
        
if st.session_state["confirm_action"] == "confirm_generate_all_skip":
    st.session_state["confirm_action"] = None
    placeholder.empty()
    skip_if_exists = True
    st.success("正在补充生成视频.................................................")
    base_info, drama_data, drama_pipeline = check_and_init(force_refresh=True, load_data=False)
    if drama_pipeline is not None:
        generate_drama_all(drama_pipeline=drama_pipeline, skip_if_exists=skip_if_exists)

# ---------------------------
# 分步生成（持久化 + 永久展示文件）
st.subheader("分步生成")
col1, col2, col3, col4, col5 = st.columns(5)

# --- Step 1: 剧本 ---
with col1:
    step1_button = st.button("生成剧本", type="primary", key="btn_script")
    del1_button  = st.button("删除剧本", type="secondary", key="del_script")

    if step1_button:
        if not api_key or not theme:
            st.warning("请填写完整配置参数（API Key、主题）")
        else:
            # _, _, drama_pipeline = check_and_init(force_refresh=False, load_data=True)
            if drama_pipeline:
                show_log()
                script_path = path_output['script'][0]
                path_drama_data = path_output['script'][1]
                results, print_text = drama_pipeline.generate_script_and_shot(skip_if_exists=True)
                with open(script_path, "r", encoding="utf-8") as f:
                    st.session_state["script_text"] = f.read()
                time.sleep(1)
                st.rerun()
    if del1_button:
        st.session_state["confirm_action"] = "delete_script"
    confirm_delete("script", session_state_key="script_text")
    # 永久展示剧本文本
    if path_output['script'] and os.path.exists(path_output['script'][0]):
        script_path = path_output['script'][0]
        with open(script_path, "r", encoding="utf-8") as f:
            script_text = f.read()
        st.markdown(script_text)

# --- Step 2: 角色 ---
with col2:
    step2_button = st.button("生成角色", type="primary", key="btn_roles")
    del2_button  = st.button("删除角色", type="secondary", key="del_roles")

    if step2_button:
        if not api_key or not theme:
            st.warning("请填写完整配置参数（API Key、主题）")
        else:
            # _, _, drama_pipeline = check_and_init()
            if drama_pipeline:
                show_log()
                results1, print_text1 = drama_pipeline.generate_voices_sync(skip_if_exists=True)
                results2, print_text2 = drama_pipeline.generate_roles(skip_if_exists=True)
                st.session_state["roles_generated"] = True
                st.session_state["roles_generated"] = True
                time.sleep(1)
                st.rerun()

    if del2_button:
        st.session_state["confirm_action"] = "delete_roles"
    confirm_delete("roles", session_state_key="roles_generated")

    # 永久展示角色图片
    for role in path_output['roles']:
        role_path = role['path']
        role_name = role.get('name', '未知角色')
        file_exists = role.get('exists', False)
        st.markdown(f"**{role_name}**")
        col_img, col_candidates = st.columns([9, 3.5])
        # with col_name:
        with col_img:
            if file_exists and os.path.exists(role_path):
                col_left, col_center, col_right = st.columns([4, 3, 3])
                with col_center:
                    if st.button("❌", key=f"del_{role_name}", help=f"删除 {role_name} 的角色图片"):
                        try:
                            os.remove(role_path)
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败: {e}")
                if file_exists and os.path.exists(role_path):
                    st.image(role_path, caption=os.path.basename(role_path), width='stretch')
            else:
                base_name = os.path.basename(role_path)
                st.markdown(f'<span style="color:red;font-weight:bold">缺失{base_name}</span>', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("⬆️上传", type=['png','jpg','jpeg', 'webp'], key=f"upload_{role_name}")
                if uploaded_file and drama_pipeline:
                    try:
                        img = Image.open(uploaded_file).convert("RGB")
                        drama_pipeline.resize_keep_aspect(
                            img,
                            save_path=role_path,
                            target_size=role_size,
                            mode="scale"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"上传失败: {e}")

        with col_candidates:
            # 候选图片：role_path_0.png ~ role_path_3.png
            base, ext = os.path.splitext(role_path)
            for i in range(4):
                candidate_path = f"{base}_{i}{ext}"
                if os.path.exists(candidate_path):
                    c_col1, c_col2 = st.columns([8, 1])
                    with c_col1:
                        st.image(candidate_path, width='stretch')
                    with c_col2:
                        if st.button("✔", key=f"select_{role_name}_{i}", help=f"选择候选图 {os.path.basename(candidate_path)} 作为 {role_name}"):
                            try:
                                img = Image.open(candidate_path).convert("RGB")
                                drama_pipeline.resize_keep_aspect(
                                    img,
                                    save_path=role_path,
                                    target_size=role_size,
                                    mode="scale"
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"替换失败: {e}")
# --- Step 3: 场景 ---
with col3:
    step3_button = st.button("生成场景", type="primary", key="btn_scenes")
    del3_button  = st.button("删除场景", type="secondary", key="del_scenes")

    if step3_button:
        if not api_key or not theme:
            st.warning("请填写完整配置参数（API Key、主题）")
        else:
            # _, _, drama_pipeline = check_and_init()
            if drama_pipeline:
                show_log()
                drama_pipeline.generate_scenes(skip_if_exists=True)
                st.session_state["scenes_generated"] = True
                time.sleep(1)
                st.rerun()
    if del3_button:
        st.session_state["confirm_action"] = "delete_scenes"
        # st.text(path_output['scenes'])
    confirm_delete("scenes", session_state_key="scenes_generated")

    # 把场景图片按场景名归类
    scene_dict = defaultdict(list)
    for scene_path in path_output['scenes']:
        # if os.path.exists(scene_path):
        base_name = os.path.splitext(os.path.basename(scene_path))[0]  # e.g. scene1_xxx
        # 提取场景名：取下划线前缀
        scene_key = base_name.split("_")[0]  
        scene_dict[scene_key].append(scene_path)

    # 展示
    for scene_name, paths in scene_dict.items():
        st.markdown(f"**{scene_name}**")
        col_left, col_right = st.columns(2)
        # for i, scene_path in enumerate(sorted(paths)):
        for i, scene_path in enumerate(paths):
            col = col_left if i % 2 == 0 else col_right
            with col:
                if os.path.exists(scene_path):
                    # 删除按钮放在图片上方
                    btn_col = st.columns([3, 3, 3])[1]  # 中间位置
                    with btn_col:
                        if st.button("❌", key=f"del_scene_{scene_name}_{i}", help=f"删除 {os.path.basename(scene_path)}"):
                            try:
                                os.remove(scene_path)
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {e}")

                    st.image(scene_path, caption=os.path.basename(scene_path), width='stretch')

                else:
                    base_name = os.path.basename(scene_path)
                    st.markdown(
                        f'<span style="color:red;font-weight:bold">缺失{base_name}</span>',
                        unsafe_allow_html=True
                    )
                    uploaded_file = st.file_uploader(
                        "⬆️上传", 
                        type=['png','jpg','jpeg', 'webp'], 
                        key=f"upload_scene_{scene_name}_{i}"
                    )
                    if uploaded_file and drama_pipeline:
                        try:
                            img = Image.open(uploaded_file).convert("RGB")
                            drama_pipeline.resize_keep_aspect(
                                img,
                                save_path=scene_path,
                                target_size=(width, height),
                                mode="scale"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"上传失败: {e}")

# --- Step 4: 视频 ---
with col4:
    step4_button = st.button("生成视频", type="primary", key="btn_videos")
    del4_button  = st.button("删除视频", type="secondary", key="del_videos")

    if step4_button:
        if not api_key or not theme:
            st.warning("请填写完整配置参数（API Key、主题）")
        else:
            if drama_pipeline:
                show_log()
                drama_pipeline.generate_videos(skip_if_exists=True)
                # drama_pipeline.concatenate_videos()
                # st.session_state["video_path"] = drama_pipeline.output_video_path
                time.sleep(1)
                st.rerun()

    if del4_button:
        st.session_state["confirm_action"] = "delete_videos"
    confirm_delete("videos", session_state_key="video_path")
    # 永久展示视频文件
    col_left, col_right = st.columns(2)
    for i, video_path in enumerate(path_output['videos']):
        col = col_left if i % 2 == 0 else col_right
        with col:
            video_name = os.path.splitext(os.path.basename(video_path))[0]  # 去掉扩展名作为场景名
            if os.path.exists(video_path):
                # st.markdown(f"**{video_name}**")
                col_0, col_1, col_2 = st.columns([4, 2, 4])
                with col_1:
                    if st.button("❌", key=f"del_{video_name}", help=f"删除 {video_name} 视频"):
                        try:
                            os.remove(video_path)
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败: {e}")
                st.video(video_path)
                st.download_button(
                    label=f"下载 {os.path.basename(video_path)}",
                    data=open(video_path, "rb").read(),
                    file_name=os.path.basename(video_path),
                )
            else:
                st.markdown(
                    f'<span style="color:red;font-weight:bold">缺失{os.path.basename(video_path)}</span>',
                    unsafe_allow_html=True
                )
                uploaded_file = st.file_uploader(
                    "⬆️上传", 
                    type=['mp4', 'mov', 'avi', 'webm'], 
                    key=f"upload_{video_name}"
                )
                if uploaded_file:
                    try:
                        with open(video_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        # st.success(f"上传成功: {os.path.basename(video_path)}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"上传失败: {e}")

# --- Step 5: 合成短剧 ---
with col5:
    step5_button = st.button("合成短剧", type="primary", key="btn_drama")
    # del5_button  = st.button("删除短剧", type="secondary", key="del_videos")
    if drama_pipeline:
        if "drama_title" not in st.session_state:
            st.session_state["drama_title"] = drama_pipeline.data.base_info['title']
        new_title = st.text_input("短剧标题", st.session_state["drama_title"])
        # 如果更新标题，则更新drama_data, drama_pipeline.data.save_json()
        if new_title != st.session_state["drama_title"]:
            st.session_state["drama_title"] = new_title
            drama_pipeline.data.base_info['title'] = new_title
            drama_pipeline.data.save_json()

    if step5_button:
        if not api_key or not theme:
            st.warning("请填写完整配置参数（API Key、主题）")
        else:
            if drama_pipeline:
                show_log()
                # drama_pipeline.generate_videos(skip_if_exists=True)
                drama_pipeline.concatenate_videos()
                st.session_state["drama_path"] = drama_pipeline.output_video_path
                time.sleep(1)
                st.rerun()

    if "drama_path" in st.session_state and st.session_state["drama_path"]:
        drama_path=st.session_state["drama_path"]
        drama_name = os.path.splitext(os.path.basename(st.session_state["drama_path"]))[0]  # 去掉扩展名作为场景名
        if os.path.exists(drama_path):
            # st.markdown(f"**{drama_name}**")
            st.video(drama_path)
            st.download_button(
                label=f"下载 {os.path.basename(drama_path)}",
                data=open(drama_path, "rb").read(),
                file_name=os.path.basename(drama_path),
            )
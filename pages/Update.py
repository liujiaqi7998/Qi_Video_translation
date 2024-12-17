import os
from pathlib import Path
import streamlit as st

from utils.file_path import PathManager

st.set_page_config(
    page_title="视频上传",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_PATH = os.path.join(BASE_DIR, "TEMP_2")
path_manager = PathManager(TEMP_PATH)

# 文件上传
file_type = ["mp4", "ts", "mov", "mxf", "mpg", "flv", "wmv", "avi", "m4v", "f4v", "mpeg", "3gp", "asf", "mkv"]


def delete_video():
    if os.path.exists(path_manager.input_video_dir):
        os.remove(path_manager.input_video_dir)
        st.success("视频已删除")
    else:
        st.warning("没有视频文件可删除")


if os.path.exists(path_manager.input_video_dir):
    video_file = open(path_manager.input_video_dir, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

    # 删除按钮
    st.button("删除当前视频", on_click=delete_video)

else:
    uploaded_file = st.file_uploader("上传视频文件", type=file_type)

    # 保存上传的文件
    if uploaded_file is not None:
        with open(path_manager.input_video_dir, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("上传成功!")

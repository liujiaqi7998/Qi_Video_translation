import os
from pathlib import Path
import streamlit as st
from utils.file_path import PathManager

st.set_page_config(
    page_title="奇慧智译",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.success("功能选择区")

BASE_DIR = Path(__file__).resolve().parent
TEMP_PATH = os.path.join(BASE_DIR, "TEMP_2")
path_manager = PathManager(TEMP_PATH)
path_manager.create_directories()

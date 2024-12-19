import gradio as gr
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config
from utils.db_utils import MainData
from utils.file_path import PathManager

css = """
#warning {background-color: #FFCCCB}
.feedback textarea {font-size: 24px !important}
"""

with gr.Blocks(css=css) as demo1:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Textbox(label="name", elem_id="warning")
            gr.Textbox(label="age", elem_classes="feedback")
        with gr.Column(scale=2):
            gr.Dropdown(["one", "two", "tree"], label="class")
            gr.CheckboxGroup(["male", "female"], label="sex")
        with gr.Column(scale=1):
            gr.Radio(["is_girl"], label="is_girl")
            gr.Slider(1, 100, 20)
    with gr.Row():
        gr.Button(value="Submit")
        gr.Button(value="Clear")

with gr.Blocks(css=css) as progress_tracking:
    import os


    def list_subdirectories(path):
        try:
            subdir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            return subdir
        except FileNotFoundError:
            return ["路径不存在"]


    def refresh_dropdown():
        dropdown.choices = list_subdirectories(config.TEMP_PATH)


    def read_sql(choose):
        path = os.path.join(config.WORKPLACE_PATH, "cache", str(choose))
        path_manager = PathManager(path)
        if not os.path.exists(path_manager.db_dir):
            raise Exception("项目进度文件不存在，请检测项目是否开始")
        engine = create_engine(f'sqlite:///{path_manager.db_dir}')
        session = None
        try:
            session = sessionmaker(bind=engine)()
            if not session.query(MainData).first():
                raise Exception("项目进度不存在，请检测项目是否开始")
            data = session.query(MainData).all()
            data_display = []
            for item in data:
                one = []
                for column in MainData.__table__.columns:
                    one.append(getattr(item, column.name))
                data_display.append(one)
            columns = [column.comment for column in MainData.__table__.columns]
            df = pd.DataFrame(data_display, columns=columns)
            return df
        finally:
            if session:
                session.close()


    sql_table = gr.Dataframe()

    with gr.Row():
        dropdown = gr.Dropdown(choices=list_subdirectories(os.path.join(config.WORKPLACE_PATH, "cache")),
                               label="项目选择")
        gr.Button(value="刷新列表", variant="primary").click(refresh_dropdown)
        gr.Button(value="读取").click(fn=read_sql, inputs=dropdown, outputs=sql_table)

app = gr.TabbedInterface([demo1, progress_tracking], ["全局预览", "进度追踪"])
app.launch(server_name='0.0.0.0', server_port=25432, show_error=True)

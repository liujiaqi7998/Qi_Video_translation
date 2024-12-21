import shutil
import sys
import uuid
from pathlib import Path
from loguru import logger
import gradio as gr
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import threading
import time
from utils.db_utils import MainData
from utils.file_path import PathManager
import os

task_list = []
BASE_DIR = Path(__file__).resolve().parent
WORKPLACE_PATH = os.environ.get("CACHE_PATH", os.path.join(BASE_DIR, "workplace"))
MAIN_PATH = os.path.join(BASE_DIR, "main.py")

# 队列数量
parallel_runs = 1
CUDA_VISIBLE_DEVICES = "1"
# 设置显卡
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

def task_scheduler():
    while True:
        for task in task_list:
            if task['status'] == '等待调度':
                # 这里可以添加任务调度的逻辑
                logger.debug(f"开始调度任务 {task['id']}")

                task_wait = 0
                for x_task in task_list:
                    if x_task['status'] == '正在处理':
                        task_wait = task_wait + 1

                if task_wait < parallel_runs:
                    logger.info(f"处理队列 {task_wait}/{parallel_runs}")
                    task['status'] = '正在处理'
                    task_folder = str(os.path.join(WORKPLACE_PATH, "cache", task["id"]))
                    log_path = os.path.join(task_folder, "log.log")
                    env_vars = f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}"
                    task_folder_little =  os.path.join("workplace", "cache", task["id"])
                    args = f"-agg {task['agg_level']} -input {task['input_language']} -output {task['output_language']} -retry {task['retry_times']} -path {task_folder_little}"
                    task['task'] = threading.Thread(target=lambda: os.system(f"{env_vars} {sys.executable} {MAIN_PATH} {args}> {log_path} 2>&1"))
                    task['task'].start()
                else:
                    logger.debug(f"处理队列已满 {task_wait}/{parallel_runs} 不处理了")
                    task['status'] = '等待调度'
            elif task['status'] == '正在处理':
                # 这里实现一下是否处理完成
                if not task['task'].is_alive():
                    task['status'] = '完成'
                    logger.info(f"任务 {task['id']} 已完成处理")

        time.sleep(0.5)


# 创建并启动任务调度线程
scheduler_thread = threading.Thread(target=task_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()

with gr.Blocks() as task_manager:
    task_table = gr.Dataframe()

    with gr.Row():
        # os.path.join(config.WORKPLACE_PATH, "input") 里面是视频文件，读取这些视频文件作为一个选择表
        video_files = [f for f in os.listdir(os.path.join(WORKPLACE_PATH, "input")) if f.endswith('.mp4')]
        video_selector = gr.Dropdown(label="选择视频文件", choices=video_files)

        # 同样的原理加一个选择字幕文件
        subtitle_files = [f for f in os.listdir(os.path.join(WORKPLACE_PATH, "input")) if f.endswith('.ass')]
        subtitle_selector = gr.Dropdown(label="选择字幕文件", choices=subtitle_files)

    # 人声提取激进程度 滑块 agg = 10 0-20 默认10
    agg_slider = gr.Slider(minimum=0, maximum=20, value=10, label="人声提取激进程度")

    with gr.Row():
        # 选择框 输入语言 input_language = "ja"
        input_language_selector = gr.Dropdown(label="输入语言", choices=["ja", "en", "zh"], value="ja")

        # 选择框 输出语言 output_language = "zh"
        output_language_selector = gr.Dropdown(label="输出语言", choices=["ja", "en", "zh"], value="zh")

    with gr.Row():
        # 整数输入框 重试次数 retry_times = 5
        retry_times_input = gr.Number(label="重试次数", value=5, precision=0)


    def read_task_table():
        data_display = []
        for item in task_list:
            item: dict
            one = []
            for k, v in item.items():
                if k != "task":
                    one.append(v)
            data_display.append(one)
        columns = ['ID', '视频文件', '字幕文件', '人声强度', '输入语言', '输出语言',
                   '重试次数', '状态']
        df = pd.DataFrame(data_display, columns=columns)
        return df


    def submit_task(video_selector, subtitle_selector, agg_slider, input_language_selector, output_language_selector,
                     retry_times_input):
        task_dict = {
            "id": str(uuid.uuid4()),
            "video_file": video_selector,
            "subtitle_file": subtitle_selector,
            "agg_level": agg_slider,
            "input_language": input_language_selector,
            "output_language": output_language_selector,
            "retry_times": retry_times_input,
            "status": "等待调度"
        }
        task_folder = os.path.join(WORKPLACE_PATH, "cache", task_dict["id"])
        os.makedirs(task_folder, exist_ok=True)
        shutil.copy(str(os.path.join(WORKPLACE_PATH, "input", task_dict["video_file"])),
                    str(os.path.join(task_folder, "input.mp4")))
        shutil.copy(str(os.path.join(WORKPLACE_PATH, "input", task_dict["subtitle_file"])),
                    str(os.path.join(task_folder, "subtitles.ass")))
        task_list.append(task_dict)
        return read_task_table()


    submit_button = gr.Button(value="提交任务")
    read_task_table_button = gr.Button(value="读取任务列表")
    submit_button.click(fn=submit_task, inputs=[video_selector, subtitle_selector, agg_slider, input_language_selector,
                                                output_language_selector, retry_times_input],
                        outputs=task_table)
    read_task_table_button.click(fn=read_task_table, outputs=task_table)

with gr.Blocks() as progress_tracking:
    def list_subdirectories(path):
        subdir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        return subdir


    def refresh_dropdown():
        new_choices = list_subdirectories(os.path.join(WORKPLACE_PATH, "cache"))
        return gr.Dropdown(choices=new_choices)


    def read_sql(choose):
        path = os.path.join(WORKPLACE_PATH, "cache", str(choose))
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
            with open(os.path.join(path, "log.log"), 'r', encoding='utf-8') as file:
                log_content = file.read()
            return df, log_content
        finally:
            if session:
                session.close()


    sql_table = gr.Dataframe()
    log_t = gr.TextArea()
    with gr.Row():
        dropdown = gr.Dropdown(choices=list_subdirectories(os.path.join(WORKPLACE_PATH, "cache")),
                               label="项目选择")
        gr.Button(value="刷新列表", variant="primary").click(fn=refresh_dropdown, outputs=dropdown)
        gr.Button(value="读取").click(fn=read_sql, inputs=dropdown, outputs=[sql_table, log_t])

app = gr.TabbedInterface([task_manager, progress_tracking], ["全局预览", "进度追踪"])
app.launch(server_name='0.0.0.0', server_port=25432, show_error=True)

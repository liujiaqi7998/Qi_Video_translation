from celery import Celery
import base64
import os
from loguru import logger
import requests
import hashlib

broker_url = os.getenv('BROKER_URL')
if not broker_url:
    raise Exception("未找到BROKER_URL环境变量，程序退出")

app = Celery('队列', broker=broker_url)

@app.task
def run_main(args):
    command = 'cd /app && /root/miniconda3/envs/Qi_Video_translation/bin/python3 main.py'
    
    logger.info("接收到任务: {}".format(args))
    
    if video_url := args.get("video_url"):
        encoded_video_name = hashlib.md5(os.path.basename(video_url).encode()).hexdigest()
        temp_dir = f'TEMP/{encoded_video_name}'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        response = requests.get(video_url)
        video_content = response.content
        with open(f'{temp_dir}/input.mp4', 'wb') as f:
            f.write(video_content)
        command += f' --temp_dir {temp_dir}'
    else:
        raise Exception("未找到video_url参数，程序退出")
    
    if subtitle_url := args.get("subtitle_url"):
        response = requests.get(subtitle_url)
        subtitle_content = response.content
        with open(f'{temp_dir}/subtitles.ass', 'wb') as f:
            f.write(subtitle_content)
    else:
        raise Exception("未找到subtitle_url参数，程序退出")
    
    if agg := args.get("agg"):
        command += f' --agg {agg}'
    
    if input_language := args.get("input_language"):
        command += f' --input_language {input_language}'
    
    if output_language := args.get("output_language"):
        command += f' --output_language {output_language}'
    
    if retry_times := args.get("retry_times"):
        command += f' --retry_times {retry_times}'
    
    
    if sub_style := args.get("sub_style"):
        command += f' --sub_style {sub_style}'
    
    if combined_mkv := args.get("combined_mkv"):
        command += f' --combined_mkv {combined_mkv}'
    
    logger.info(f"运行项目：{command}")
    
    os.system(command)

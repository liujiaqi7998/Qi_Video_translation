import os
import time
from loguru import logger
from boto3.session import Session
from botocore.client import Config
import hashlib

import requests

manager_url = os.getenv('MANAGER_URL')
endpoint_url = os.getenv('S3_ENDPOINT')
access_key = os.getenv('S3_AK')
secret_key = os.getenv('S3_SK')
bucket = os.getenv('S3_BUCKET')

if not manager_url:
    raise Exception("未找到MANAGER_URL环境变量，程序退出")

if not endpoint_url:
    raise Exception("未找到S3_ENDPOINT环境变量，程序退出")

if not access_key:
    raise Exception("未找到S3_AK环境变量，程序退出")

if not secret_key:
    raise Exception("未找到S3_SK环境变量，程序退出")

if not bucket:
    raise Exception("未找到S3_BUCKET环境变量，程序退出")

if not os.path.exists('TEMP'):
    os.makedirs('TEMP')
    

session = Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)

# 连接到minio
s3_client = session.resource('s3', endpoint_url=endpoint_url, config=Config(s3={'addressing_style': 'path'}))


def submit_log(url_log, task_id, log):
    requests.post(url_log, json={"task_id": task_id, "log": log})


def run_main(args):
    command = 'cd /app && /root/miniconda3/envs/Qi_Video_translation/bin/python3 main.py'
    
    logger.info("接收到任务: {}".format(args))
    submit_log(args.get("url_log"), args.get("task_id"), "接收到任务: {}".format(args))
    
    if video_key := args.get("video_key"):
        temp_dir = f'TEMP/{hashlib.md5(video_key.encode()).hexdigest()}'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        s3_client.Object(bucket, video_key).download_file(f'{temp_dir}/input.mp4')
        command += f' --temp_dir {temp_dir}'
    else:
        raise Exception("未找到video_key参数，程序退出")
    
    if subtitle_key := args.get("subtitle_key"):
        s3_client.Object(bucket, subtitle_key).download_file(f'{temp_dir}/subtitles.ass')
    else:
        raise Exception("未找到subtitle_key参数，程序退出")
    
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
    submit_log(args.get("url_log"), args.get("task_id"), f"运行项目：{command}")

    # 运行项目
    os.system(command)
    
    # 生成的视频是output.mkv
    output_key = args.get("output_key")
    if not output_key:
        raise Exception("未找到output_key参数，程序退出")
    # 判断视频是否存在
    if not os.path.exists(f'{temp_dir}/output.mkv'):
        raise Exception("未找到output.mkv文件，程序退出")
    s3_client.Object(bucket, output_key).upload_file(f'{temp_dir}/output.mkv')

    logger.success("任务已完成")
    submit_log(args.get("url_log"), args.get("task_id"), "任务已完成")


def run_get_task():
    url_get = f"{manager_url}/qi_video_translation/get_video_translation_undone_task/"
    url_change = f"{manager_url}/qi_video_translation/change_video_translation_task/"
    url_log = f"{manager_url}/qi_video_translation/submit_video_translation_log/"
    response = requests.get(url_get)
    if response.status_code == 200:
        data = response.json().get("data")
        if data:
            logger.success(f"拉取任务成功: {data}")
            # 这里可以添加处理data的逻辑
            try:
                run_main({
                    "video_key": data["video_path"],
                    "subtitle_key": data["subtitle_path"],
                    "input_language": data["source_language"],
                    "output_language": data["target_language"],
                    "output_key": data["output_path"],
                    "sub_style": data["sub_style"],
                    "agg": data["agg"],
                    "retry_times": data["retry_times"],
                    "url_log": url_log,
                })
                requests.post(url_change, json={"task_id": data["id"], "status": "完成"})
            except Exception as e:
                logger.error(f"任务处理失败: {e}")
                requests.post(url_log, json={"task_id": data["id"], "log": str(e)})
                requests.post(url_change, json={"task_id": data["id"], "status": "失败"})
        else:
            logger.info("获取任务成功，但未找到数据")
    else:
        logger.error(f"获取任务失败，状态码: {response.status_code}, 错误信息: {response.text}")


if __name__ == "__main__":
    while True:
        try:
            run_get_task()
        except Exception as e:
            logger.error(f"任务处理失败: {e}")
        time.sleep(1)
from celery import Celery
import os
from loguru import logger
from boto3.session import Session
import hashlib
from botocore.client import Config

broker_url = os.getenv('BROKER_URL')
endpoint_url = os.getenv('S3_ENDPOINT')
access_key = os.getenv('S3_AK')
secret_key = os.getenv('S3_SK')
bucket = os.getenv('S3_BUCKET')

if not broker_url:
    raise Exception("未找到BROKER_URL环境变量，程序退出")

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
        
app = Celery('队列', broker=broker_url)

session = Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)

# 连接到minio
s3_client = session.resource('s3', endpoint_url=endpoint_url, config=Config(s3={'addressing_style': 'path'}))


@app.task
def run_main(args):
    command = 'cd /app && /root/miniconda3/envs/Qi_Video_translation/bin/python3 main.py'
    
    logger.info("接收到任务: {}".format(args))
    
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
    
    os.system(command)
    
    # 生成的视频是output.mp4
    output_key = args.get("output_key")
    if not output_key:
        raise Exception("未找到output_key参数，程序退出")
    # 判断视频是否存在
    if not os.path.exists(f'{temp_dir}/output.mkv'):
        raise Exception("未找到output.mp4文件，程序退出")
    s3_client.Object(bucket, output_key).upload_file(f'{temp_dir}/output.mkv')

    logger.info("任务已完成")


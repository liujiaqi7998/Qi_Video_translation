from celery import Celery

import os

broker_url = os.getenv('BROKER_URL')
if not broker_url:
    raise Exception("未找到BROKER_URL环境变量，程序退出")

app = Celery('tasks', broker=broker_url)

@app.task
def run_main(args):
    command = 'python main.py'
    
    if agg := args.get("agg"):
        command += f'--agg {agg}'
    
    if input_language := args.get("input_language"):
        command += f' --input_language {input_language}'
    
    if output_language := args.get("output_language"):
        command += f' --output_language {output_language}'
    
    if retry_times := args.get("retry_times"):
        command += f' --retry_times {retry_times}'
    
    if temp_dir := args.get("temp_dir"):
        command += f' --temp_dir {temp_dir}'
    
    if sub_style := args.get("sub_style"):
        command += f' --sub_style {sub_style}'
    
    if combined_mkv := args.get("combined_mkv"):
        command += f' --combined_mkv {combined_mkv}'
    
    print(f"运行项目：{command}")
    os.system(command)

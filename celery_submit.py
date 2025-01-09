from celery_work import run_main
from loguru import logger

# 如果不知道参数，就别加
args = {
    "input_language": "ja",
    "output_language": "zh",
    "video_key": "input_video/14.mp4",
    "subtitle_key": "input_ass/14.ass",
    "output_key": "output_video/14.mp4"
}
task = run_main.apply_async(args=[args])
logger.info(f"任务{task.id}已提交")

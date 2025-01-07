from celery_work import run_main
from loguru import logger

# 如果不知道参数，就别加
args = {
    "input_language": "ja",
    "output_language": "zh",
    "video_key": "video/[DMG&RoxyLib] 無職転生 第12話「魔眼を持つ女」 [BDRip][AVC_AAC][1080P][CHS](5BDF26F5).mp4",
    "subtitle_key": "ass/12.ass",
    "output_key": "output/12.mp4"
}
task = run_main.apply_async(args=[args])
logger.info(f"任务{task.id}已提交")

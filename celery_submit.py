from celery_work import run_main

# 如果不知道参数，就别加
args = {
    "input_language": "ja",
    "output_language": "zh",
    "video_url": "http://192.168.1.11:19091/qi-video-translation/video/[DMG&RoxyLib] 無職転生 第06話「ロアの休日」 [BDRip][AVC_AAC][1080P][CHS](B0C74FEF).mp4",
    "subtitle_url": "http://192.168.1.11:19091/qi-video-translation/ass/6.ass",
}

task = run_main.delay(args)

print(task.id)



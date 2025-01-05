import os

import ffmpeg
from loguru import logger
from sqlalchemy import create_engine

import config

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.description = '请指定先关参数'

    TEMP_PATH = os.path.join(config.BASE_DIR, "TEMP")

    parser.add_argument("-agg", "--agg", help="人声提取激进程度,0-20，默认10", dest="agg", type=int, default="10")
    parser.add_argument("-input", "--input_language", help="输入语言,默认ja", dest="input_language", type=str, default="ja")
    parser.add_argument("-output", "--output_language", help="输出语言,默认zh", dest="output_language", type=str, default="zh")
    parser.add_argument("-retry", "--retry_times", help="重试次数,默认重试5次", dest="retry_times", type=int, default="5")
    parser.add_argument("-path", "--temp_dir", help="工作目录,默认./TEMP", dest="TEMP_PATH", type=str, default=TEMP_PATH)
    parser.add_argument("-sub_style", "--sub_style", help="字幕用于处理的目标style，默认会通过筛选出场率最高的style作为目标处理字幕", dest="sub_style", type=str, default="")
    parser.add_argument("-mkv", "--combined_mkv", help="合成视频成mkv文件", dest="combined_mkv", type=int, default="1")
    args = parser.parse_args()

    config.agg = args.agg
    config.output_language = args.output_language
    config.input_language = args.input_language
    config.retry_times = args.retry_times
    config.TEMP_PATH = args.TEMP_PATH
    config.sub_style = args.sub_style
    config.combined_mkv = args.combined_mkv

from utils.cut_video import CutVideo
from utils.db_utils import Base
from utils.file_path import PathManager
from utils.mix_voice import MixVoice
from utils.optimization_task import OptimizationTask
from utils.out_put_task import OutPutTask
from utils.speaker_asr import SpeakerASR
from utils.speaker_separation import deal_uvr_all_video, SpeakerSeparation
from utils.subtitles_extraction import SubtitlesExtraction
from utils.tts_task import TTSTask


def init_video(path_manager: PathManager):
    # Fix: 判断一下是否已经存在，存在就不再转换
    if not os.path.exists(path_manager.input_voice_dir):
        (ffmpeg.input(path_manager.input_video_dir).
         output(path_manager.input_voice_dir, channels=2, sample_rate=44100).
         run(overwrite_output=True))

    if not os.path.exists(path_manager.input_voice_dir):
        raise FileNotFoundError(f"指定的音频文件不存在: {path_manager.input_voice_dir}，可能是ffmpeg提取视频音频失败")


def main():

    path_manager = PathManager(config.TEMP_PATH)
    path_manager.create_directories()
    engine = create_engine(f'sqlite:///{path_manager.db_dir}')

    if not os.path.exists(path_manager.db_dir):
        # 这里实现一下创建数据库
        Base.metadata.create_all(engine)

    # 第一步，利用ffmpeg将视频的音频提取出来
    logger.info("ffmpeg，提取视频音频开始")
    init_video(path_manager)
    logger.info("ffmpeg，提取视频音频结束")

    # 字幕分割音频
    subtitles_extraction = SubtitlesExtraction(engine, path_manager)
    subtitles_extraction.main()

    # 视频切割
    cut_video = CutVideo(engine, path_manager)
    cut_video.main()

    # 提取视频背景音频
    deal_uvr_all_video(path_manager)

    # 视频提取人声
    speaker_separation = SpeakerSeparation(engine, path_manager)
    speaker_separation.main()

    # ASR 识别
    speaker_asr = SpeakerASR(engine, path_manager)
    speaker_asr.main()

    # TTS人声开始
    tts_task = TTSTask(engine, path_manager)
    tts_task.main()

    # TTS高清修复
    tts_task_fix = OptimizationTask(engine, path_manager)
    tts_task_fix.main()

    # 混合音频
    mix_voice = MixVoice(engine, path_manager)
    mix_voice.main()

    # 输出
    out_put = OutPutTask(engine, path_manager)
    out_put.main()


if __name__ == '__main__':
    main()

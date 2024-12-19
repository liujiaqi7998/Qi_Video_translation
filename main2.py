import os

import ffmpeg
from loguru import logger
from sqlalchemy import create_engine

import config
from utils.cut_video import CutVideo
from utils.db_utils import Base
from utils.file_path import PathManager
from utils.mix_voice import MixVoice
from utils.optimization_task import OptimizationTask
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

    # logger.info("开始合并输出音频")
    #
    # input_voice_audio = AudioSegment.from_file(path_manager.input_voice_dir)

    # for id, subtitle in subtitles.items():
    #     if not os.path.exists(os.path.join(path_manager.cut_mix_dir, f'{id}.wav')):
    #         continue
    #     this_pic = AudioSegment.from_file(os.path.join(path_manager.cut_mix_dir, f'{id}.wav'))
    #     start = subtitle.get("start")
    #     end = subtitle.get("end")
    #     input_voice_audio = input_voice_audio[:start] + this_pic + input_voice_audio[end:]
    #
    # input_voice_audio.export(path_manager.output_voice_dir, format='wav')
    # logger.info("音频合成完成")
    #
    # input_voice_audio.export(path_manager.output_voice_mp3_dir, format='mp3', bitrate="192k")
    # logger.info("输出音频已压缩为MP3格式，方便传输测试")
    #


if __name__ == '__main__':
    main()

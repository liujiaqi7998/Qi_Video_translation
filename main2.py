import json
import os
import re
import shutil
import tempfile
from loguru import logger
import pysubs2
from retrying import retry
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils.cut_video import CutVideo
from utils.db_utils import Base, MainData
from utils.file_path import PathManager
from utils.speaker_separation import deal_uvr_all_video, SpeakerSeparation
from utils.subtitles_extraction import SubtitlesExtraction

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
import langid
import ffmpeg
import torch
import whisper
from pydub import AudioSegment
import config
from tools.uvr5.vr import AudioPre

BASE_DIR = Path(__file__).resolve().parent
TEMP_PATH = os.path.join(BASE_DIR, "TEMP_2")
sys.path.append(f"{BASE_DIR}")
sys.path.append(f"{BASE_DIR}/GPT_SoVITS")
import soundfile as sf
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names

# 人声提取激进程度 0-20，默认10
agg = 10
input_language = "ja"
output_language = "zh"
# 重试次数
retry_times = 5
# log_level = 'DEBUG'
log_level = 'INFO'

cut_method_names = get_cut_method_names()
config_path = os.path.join(BASE_DIR, "GPT_SoVITS/configs/tts_infer.yaml")
tts_config = TTS_Config(config_path)
logger.debug(f"\n{tts_config}")
tts_pipeline = TTS(tts_config)

logger.info(f"程序启动成功，运行目录是：{BASE_DIR}")


def init_video(path_manager: PathManager):
    # Fix: 判断一下是否已经存在，存在就不再转换
    if not os.path.exists(path_manager.input_voice_dir):
        (ffmpeg.input(path_manager.input_video_dir).
         output(path_manager.input_voice_dir, channels=2, sample_rate=44100).
         run(overwrite_output=True))

    if not os.path.exists(path_manager.input_voice_dir):
        raise FileNotFoundError(f"指定的音频文件不存在: {path_manager.input_voice_dir}，可能是ffmpeg提取视频音频失败")


def deal_uvr_video(path_manager: PathManager, subtitles: dict):
    try:
        if not os.path.exists(path_manager.instrument_dir):
            raise Exception("没有有效的背景音频文件，可能是上一步人声分离失败")

        instrument_audio = AudioSegment.from_file(path_manager.instrument_dir)

        # 这里进行人声处理
        with tempfile.TemporaryDirectory() as tmp:
            count = 0
            for id, subtitle in subtitles.items():
                vocal_path = os.path.join(path_manager.cut_asr_vocal_dir, f"{id}.wav")
                raw_path = os.path.join(path_manager.cut_asr_raw_dir, f"{id}.wav")
                if os.path.exists(vocal_path):
                    continue
                if not os.path.exists(raw_path):
                    continue
                shutil.copyfile(raw_path, os.path.join(tmp, f"{id}.wav"))
                count = count + 1
            logger.info(f"一共{count}个音频需要提取人声")
            if count > 0:
                os.system(f"{config.resemble_enhance_cmd} {tmp} {path_manager.cut_asr_vocal_dir}")
            logger.info(f"人声提取完成")
            # 背景音频裁切
            for id, subtitle in subtitles.items():
                instrument_path = os.path.join(path_manager.cut_instrument_dir, f"{id}.wav")
                if os.path.exists(instrument_path):
                    continue
                instrument_audio[subtitle.get('start'):subtitle.get('end')].export(instrument_path)
                if os.path.exists(instrument_path):
                    subtitles[id]["is_uvr"] = True

    except Exception as err:
        logger.error(err)
    finally:
        logger.info("释放torch.cuda")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return subtitles


def deal_asr(path_manager: PathManager, subtitles: dict):
    asr_model = None
    try:
        asr_model = whisper.load_model("large-v3",
                                       download_root=os.path.join(BASE_DIR, config.asr_models_path, "Whisper-large-v3"))

        for id, subtitle in subtitles.items():

            if subtitles[id].get("asr_result"):
                continue

            vocal_path = os.path.join(path_manager.cut_asr_vocal_dir, f"{id}.wav")

            if subtitle.get('end') - subtitle.get('start') > 9000:
                logger.info(f"{subtitle.get('id')}: {subtitle.get('text')} 大于9秒，强制9秒截断")
                audio_segment = AudioSegment.from_wav(vocal_path)
                vocal_path = os.path.join(path_manager.cut_asr_vocal_dir, f"{id}_9s.wav")
                audio_segment[subtitle.get('start'):subtitle.get('start') + 9000].export(vocal_path)
                subtitles[id]["out_9s"] = True

            if not os.path.exists(vocal_path):
                continue

            @retry(stop_max_attempt_number=retry_times)
            def need_retry():
                try:
                    logger.info(f"【ASR人声处理】{subtitle.get('id')}: {subtitle.get('text')}")
                    asr_result = asr_model.transcribe(vocal_path, language=input_language, initial_prompt=None)
                    if not (input_language in asr_result.get('language')):
                        logger.info(
                            f"检测到的语言与目标语言不一致：{subtitle.get('id')}: {subtitle.get('text')}: {asr_result.get('language')}")
                        return
                    subtitles[id]["asr_result"] = asr_result
                except Exception as err:
                    logger.warning(f"处理{subtitle.get('id')}: {subtitle.get('text')} 发生异常:{err}，触发重试")
                    raise err

            try:
                need_retry()
            except Exception as err:
                logger.error(f"处理{subtitle.get('id')}: {subtitle.get('text')} 发生异常:{err}")
                continue
    except Exception as err:
        logger.error(err)
    finally:
        try:
            if asr_model:
                del asr_model
        except Exception as err:
            logger.error(f"asr异常：{err}")
        logger.info("释放torch.cuda")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return subtitles


def deal_tts(path_manager: PathManager, subtitles: dict):
    result_total = {
        "success": [],
        "failed": [],
        "out_of_time": []
    }
    x = 0
    subtitles_items = list(subtitles.items())
    for id, subtitle in subtitles_items:
        if subtitle.get("is_tts"):
            continue

        if asr_result := subtitle.get('asr_result'):

            ref_audio_path = os.path.join(path_manager.cut_asr_vocal_dir, f"{id}.wav")

            if subtitle.get("out_9s"):
                # 超过9秒，使用9秒裁剪音频
                ref_audio_path = os.path.join(path_manager.cut_asr_vocal_dir, f"{id}_9s.wav")

            if not os.path.exists(ref_audio_path):
                # 解决 ref_audio_path 音频不存在问题
                continue

            prompt_text = asr_result.get('text')

            req = {
                "text": subtitle.get("text"),  # str.(required) text to be synthesized
                "text_lang": output_language,  # str.(required) language of the text to be synthesized
                "ref_audio_path": ref_audio_path,  # str.(required) reference audio path
                "aux_ref_audio_paths": [],
                # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                "prompt_text": prompt_text,  # str.(optional) prompt text for the reference audio
                "prompt_lang": asr_result.get('language'),
                # str.(required) language of the prompt text for the reference audio
                "top_k": 5,  # int. top k sampling
                "top_p": 1,  # float. top p sampling
                "temperature": 1,  # float. temperature for sampling
                "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,  # int. batch size for inference
                "batch_threshold": 0.75,  # float. threshold for batch splitting.
                "split_bucket": True,  # bool. whether to split the batch into multiple buckets.
                "return_fragment": False,  # bool. step by step return the audio fragment.
                "speed_factor": 1.0,  # float. control the speed of the synthesized audio.
                "fragment_interval": 0.3,  # float. to control the interval of the audio fragment.
                "seed": -1,  # int. random seed for reproducibility.
                "parallel_infer": True,  # bool. whether to use parallel inference.
                "repetition_penalty": 1.35  # float. repetition penalty for T2S model.
            }

            @retry(stop_max_attempt_number=retry_times)
            def need_retry():
                try:
                    output_wav = os.path.join(path_manager.cut_tts_dir, f'{id}.wav')
                    if os.path.exists(output_wav):
                        audio_segment = AudioSegment.from_wav(output_wav)
                        duration_ms = len(audio_segment)
                        cost_time = (subtitle.get("end") - subtitle.get("start"))
                        if duration_ms > cost_time:
                            # 判断出来确实有必要修改一下配音速度，否则就直接跳过这个
                            speed = (duration_ms / cost_time) + 0.1
                            logger.info(f"检测到音频存在，压缩时长到{speed}")
                            req["speed_factor"] = speed
                        else:
                            return
                    tts_generator = tts_pipeline.run(req)
                    sr, audio_data = next(tts_generator)
                    sf.write(output_wav, audio_data, sr, format='wav')
                    audio_segment = AudioSegment.from_wav(output_wav)
                    duration_ms = len(audio_segment)
                    cost_time = (subtitle.get("end") - subtitle.get("start"))
                    if duration_ms > cost_time:
                        result_total["out_of_time"].append(id)
                        raise Exception("输出音频时长大于原音频，需要压缩时长")
                    result_total["success"].append(id)
                except Exception as err:
                    logger.warning(f"处理{subtitle.get('id')}: {subtitle.get('text')} 发生异常:{err}，触发重试")
                    raise err

            try:
                need_retry()
            except Exception as e:
                logger.error(f"TTS异常:{e}")
        x = x + 1
    logger.info(
        f"""音频合成完成：【成功】{len(result_total["success"])}个 |【失败】{len(result_total["failed"])}个 |【压缩时间】{len(result_total["out_of_time"])}个""")


def deal_mix_voice(path_manager: PathManager, subtitles: dict):
    for id, subtitle in subtitles.items():
        translated_wav = os.path.join(path_manager.cut_fix_dir, f'{id}.wav')
        instrument_wav = os.path.join(path_manager.cut_instrument_dir, f'{id}.wav')
        output_wav = os.path.join(path_manager.cut_mix_dir, f'{id}.wav')
        if os.path.exists(output_wav):
            continue
        if (not os.path.exists(translated_wav)) or (not os.path.exists(instrument_wav)):
            continue
        translated_audio = AudioSegment.from_file(translated_wav)
        background_audio = AudioSegment.from_file(instrument_wav)
        mixed_audio = background_audio.overlay(translated_audio)
        mixed_audio.export(output_wav, format="wav")
        pass


def deal_fix_video(path_manager: PathManager, subtitles: dict):
    try:
        # 这里进行人声处理
        with tempfile.TemporaryDirectory() as tmp:
            count = 0
            for id, subtitle in subtitles.items():
                tts_path = os.path.join(path_manager.cut_tts_dir, f"{id}.wav")
                if not os.path.exists(tts_path):
                    continue
                fix_path = os.path.join(path_manager.cut_fix_dir, f"{id}.wav")
                if os.path.exists(fix_path):
                    continue
                shutil.copyfile(tts_path, os.path.join(tmp, f"{id}.wav"))
                count = count + 1
            logger.info(f"一共{count}个音频需要提取人声")
            if count > 0:
                os.system(f"{config.resemble_enhance_cmd} {tmp} {path_manager.cut_fix_dir}")
            logger.info(f"语音修复完成")
    except Exception as err:
        logger.error(err)
    finally:
        logger.info("释放torch.cuda")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return subtitles


def main():
    path_manager = PathManager(TEMP_PATH)
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

    # logger.info("字幕处理开始")
    # deal_subtitles(engine, path_manager)
    # logger.info("字幕处理结束")
    #
    # logger.info("音频与字幕剪切开始")
    # deal_cut_video(engine, path_manager)
    # logger.info("音频与字幕剪切完成")
    #
    # logger.info("音频分离人声开始")

    # subtitles = deal_uvr_video(path_manager, subtitles)
    # with open(path_manager.subtitles_result_dir, "w", encoding="utf-8") as f:
    #     json.dump(subtitles, f, ensure_ascii=False, indent=4)
    # logger.info("音频分离人声完成")
    #
    # logger.info("ASR人声开始")
    # subtitles = deal_asr(path_manager, subtitles)
    # with open(path_manager.subtitles_result_dir, "w", encoding="utf-8") as f:
    #     json.dump(subtitles, f, ensure_ascii=False, indent=4)
    # logger.info("ASR人声完成")
    #
    # logger.info("TTS人声开始")
    # deal_tts(path_manager, subtitles)
    # logger.info("TTS人声结束")
    #
    # logger.info("高清修复开始")
    # deal_fix_video(path_manager, subtitles)
    # logger.info("高清修复结束")
    #
    # logger.info("开始混音")
    # deal_mix_voice(path_manager, subtitles)
    # logger.info("混音结束")
    #
    # logger.info("开始合并输出音频")
    #
    # input_voice_audio = AudioSegment.from_file(path_manager.input_voice_dir)
    #
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

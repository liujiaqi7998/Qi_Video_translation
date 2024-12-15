import json
import logging
import os
import re
from collections import deque
import librosa
import pysubs2
# from modelscope import Tasks as modelscope_Tasks
from retrying import retry
# from modelscope.pipelines import pipeline as modelscope_pipeline
from utils.file_path import PathManager

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

import langid
from langdetect import detect_langs
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
from pyannote.audio import Pipeline as pyannote_Pipeline

pyannote_pipeline = pyannote_Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                      use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

# 人声提取激进程度 0-20，默认10
agg = 10
input_language = "ja"
output_language = "zh"
# 重试次数
retry_times = 5
# log_level = 'DEBUG'
log_level = 'INFO'

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=log_level)

cut_method_names = get_cut_method_names()
config_path = os.path.join(BASE_DIR, "GPT_SoVITS/configs/tts_infer.yaml")
tts_config = TTS_Config(config_path)
logging.debug(f"\n{tts_config}")
tts_pipeline = TTS(tts_config)

logging.info(f"程序启动成功，运行目录是：{BASE_DIR}")


def init_video(path_manager: PathManager):
    # Fix: 判断一下是否已经存在，存在就不再转换
    if not os.path.exists(path_manager.input_voice_dir):
        (ffmpeg.input(path_manager.input_video_dir).
         output(path_manager.input_voice_dir, channels=2, sample_rate=44100).
         run(overwrite_output=True))

    if not os.path.exists(path_manager.input_voice_dir):
        raise FileNotFoundError(f"指定的音频文件不存在: {path_manager.input_voice_dir}，可能是ffmpeg提取视频音频失败")


def deal_subtitles(path_manager: PathManager):
    use_subs_style = {}
    subs = pysubs2.load(path_manager.subtitles_dir)
    if not subs:
        raise Exception("字幕文件不生效")
    for event in subs.events:
        if event.plaintext:
            language = langid.classify(event.plaintext)
            if not language:
                continue
            if not (output_language in language[0]):
                logging.info(f"检测到的语言与目标语言不一致：{event.plaintext}")
                continue
            if not use_subs_style.get(event.style):
                use_subs_style[event.style] = 1
            else:
                use_subs_style[event.style] = use_subs_style[event.style] + 1
        pass
    # 利用 max() 函数找到得分最高的项
    most_use_style = max(use_subs_style, key=use_subs_style.get)
    logging.info(f"字幕得使用率最高的style是: {most_use_style}，得分为: {use_subs_style[most_use_style]}")
    subtitles = {}
    start_id = 0
    for event in subs.events:
        if event.plaintext and event.style == most_use_style:
            pattern = r"(www.|http|字幕组)"
            if re.search(pattern, event.plaintext):
                logging.info(f"无意义字幕组: {event.plaintext}")
                continue
            # 解决一下字幕包含括号注解问题，如果发现这个字幕完全是个注释那么直接跳过处理
            event.plaintext = re.sub(r'\(.*?\)', '', event.plaintext)
            if len(event.plaintext) <= 0:
                continue
            subtitles[start_id] = {
                "id": start_id,
                "start": event.start,
                "end": event.end,
                "text": event.plaintext
            }
            start_id = start_id + 1
    return subtitles


def deal_cut_video(path_manager: PathManager, subtitles: dict):
    input_wav_audio = AudioSegment.from_file(file=path_manager.input_voice_dir)
    for id, subtitle in subtitles.items():
        output_path = os.path.join(path_manager.cut_asr_raw_dir, f"{id}.wav")
        if not os.path.exists(output_path):
            if subtitle.get('end') - subtitle.get('start') < 1000:
                logging.info(f"{subtitle.get('id')}: {subtitle.get('text')} 小于1秒，忽略采集")
                continue
            input_wav_audio[subtitle.get('start'):subtitle.get('end')].export(output_path, format='wav')
        subtitles[id]["is_cut"] = True
        pass
    return subtitles


def deal_uvr_video(path_manager: PathManager, subtitles: dict):
    pre_fun = None
    try:
        pre_fun = AudioPre(
            agg=agg,
            model_path=os.path.join(BASE_DIR, config.uvr5_weights_path, "5_HP-Karaoke-UVR.pth"),
            device=config.infer_device,
            is_half=config.is_half,
        )

        # ans = modelscope_pipeline(
        #     modelscope_Tasks.acoustic_noise_suppression,
        #     model='damo/speech_frcrn_ans_cirm_16k')

        for id, subtitle in subtitles.items():
            input_path = os.path.join(path_manager.cut_asr_raw_dir, f"{id}.wav")
            vocal_path = os.path.join(path_manager.cut_asr_vocal_dir, f"vocal_{id}.wav_10.wav")
            instrument_path = os.path.join(path_manager.cut_instrument_dir, f"instrument_{id}.wav_10.wav")
            if not os.path.exists(input_path):
                continue
            if os.path.exists(vocal_path) and os.path.exists(instrument_path):
                continue

            @retry(stop_max_attempt_number=retry_times)
            def need_retry():
                try:
                    logging.info(f"【提取人声处理】{subtitle.get('id')}: {subtitle.get('text')}")
                    pre_fun._path_audio_(
                        music_file=input_path,
                        ins_root=path_manager.cut_instrument_dir,
                        vocal_root=path_manager.cut_asr_vocal_dir,
                        format="wav",
                        is_hp3=False
                    )
                    if (not os.path.exists(vocal_path)) or (not os.path.exists(instrument_path)):
                        raise Exception("人声分离没有产出文件")

                except Exception as err:
                    logging.warning(f"处理{subtitle.get('id')}: {subtitle.get('text')} 发生异常:{err}，触发重试")
                    raise err

            try:
                need_retry()
            except Exception as err:
                logging.error(f"处理{subtitle.get('id')}: {subtitle.get('text')} 发生异常:{err}")
                continue

            subtitles[id]["is_uvr"] = True

    except Exception as err:
        logging.error(err)
    finally:
        try:
            if pre_fun:
                del pre_fun.model
                del pre_fun
        except Exception as err:
            logging.error(f"uvr5异常：{err}")
        logging.info("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return subtitles


def deal_asr(path_manager: PathManager, subtitles: dict):
    asr_model = None
    try:
        asr_model = whisper.load_model("large-v3",
                                       download_root=os.path.join(BASE_DIR, config.asr_models_path, "Whisper-large-v3"))

        for _ in range(0, retry_times):
            # 循环n遍，太离谱了，居然是运气模式！没事多重试几次就好了，哈哈哈^_^！
            for id, subtitle in subtitles.items():

                if subtitles[id].get("asr_result"):
                    continue

                vocal_path = os.path.join(path_manager.cut_asr_vocal_dir, f"vocal_{id}.wav_10.wav")

                if subtitle.get('end') - subtitle.get('start') > 9000:
                    logging.info(f"{subtitle.get('id')}: {subtitle.get('text')} 大于9秒，强制9秒截断")
                    audio_segment = AudioSegment.from_wav(vocal_path)
                    vocal_path = os.path.join(path_manager.cut_asr_vocal_dir, f"vocal_{id}_9s.wav_10.wav")
                    audio_segment[subtitle.get('start'):subtitle.get('start') + 9000].export(vocal_path)
                    subtitles[id]["out_9s"] = True

                if not os.path.exists(vocal_path):
                    continue

                @retry(stop_max_attempt_number=retry_times)
                def need_retry():
                    try:
                        logging.info(f"【ASR人声处理】{subtitle.get('id')}: {subtitle.get('text')}")
                        asr_result = asr_model.transcribe(vocal_path, language=input_language, initial_prompt=None)
                        if not (input_language in asr_result.get('language')):
                            logging.info(
                                f"检测到的语言与目标语言不一致：{subtitle.get('id')}: {subtitle.get('text')}: {asr_result.get('language')}")
                            return
                        subtitles[id]["asr_result"] = asr_result
                    except Exception as err:
                        logging.warning(f"处理{subtitle.get('id')}: {subtitle.get('text')} 发生异常:{err}，触发重试")
                        raise err

                try:
                    need_retry()
                except Exception as err:
                    logging.error(f"处理{subtitle.get('id')}: {subtitle.get('text')} 发生异常:{err}")
                    continue
    except Exception as err:
        logging.error(err)
    finally:
        try:
            if asr_model:
                del asr_model
        except Exception as err:
            logging.error(f"asr异常：{err}")
        logging.info("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return subtitles


def deal_tts(path_manager: PathManager, subtitles: dict):
    result_total = {
        "success": [],
        "failed": [],
        "out_of_time": []
    }
    for _ in range(0, retry_times):
        for id, subtitle in subtitles.items():
            if subtitle.get("is_tts"):
                continue

            if asr_result := subtitle.get('asr_result'):

                ref_audio_path = os.path.join(path_manager.cut_asr_vocal_dir, f"vocal_{id}.wav_10.wav")

                if subtitle.get("out_9s"):
                    # 超过9秒，使用9秒裁剪音频
                    ref_audio_path = os.path.join(path_manager.cut_asr_vocal_dir, f"vocal_{id}_9s.wav_10.wav")

                if not os.path.exists(ref_audio_path):
                    # 解决 ref_audio_path 音频不存在问题
                    continue

                req = {
                    "text": subtitle.get("text"),  # str.(required) text to be synthesized
                    "text_lang": output_language,  # str.(required) language of the text to be synthesized
                    "ref_audio_path": ref_audio_path,  # str.(required) reference audio path
                    "aux_ref_audio_paths": [],
                    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                    "prompt_text": asr_result.get('text'),  # str.(optional) prompt text for the reference audio
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
                        output_wav = os.path.join(path_manager.translated_vocal_dir, f'{id}.wav')
                        if os.path.exists(output_wav):
                            audio_segment = AudioSegment.from_wav(output_wav)
                            duration_ms = len(audio_segment)
                            cost_time = (subtitle.get("end") - subtitle.get("start"))
                            if duration_ms > cost_time:
                                # 判断出来确实有必要修改一下配音速度，否则就直接跳过这个
                                speed = (duration_ms / cost_time) + 0.1
                                logging.info(f"检测到音频存在，压缩时长到{speed}")
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
                            raise Exception("输出音频时长大于原音频，需要压缩时长")
                    except Exception as err:
                        logging.warning(f"处理{subtitle.get('id')}: {subtitle.get('text')} 发生异常:{err}，触发重试")
                        raise err
                try:
                    need_retry()
                except Exception as e:
                    logging.error(f"TTS异常:{e}")

    logging.info(
        f"""音频合成完成：【成功】{len(result_total["success"])}个 |【失败】{len(result_total["failed"])}个 |【压缩时间】{len(result_total["out_of_time"])}个""")


def main():
    path_manager = PathManager(TEMP_PATH)
    path_manager.create_directories()

    # 第一步，利用ffmpeg将视频的音频提取出来
    logging.info("ffmpeg，提取视频音频开始")
    init_video(path_manager)
    logging.info("ffmpeg，提取视频音频结束")

    logging.info("字幕处理开始")
    subtitles = {}
    if not os.path.exists(path_manager.subtitles_result_dir):
        subtitles = deal_subtitles(path_manager)
        if not subtitles:
            raise Exception("字幕提取环节出现了错误导致没有字幕输出")
        with open(path_manager.subtitles_result_dir, "w", encoding="utf-8") as f:
            json.dump(subtitles, f, ensure_ascii=False, indent=4)
    else:
        with open(path_manager.subtitles_result_dir, "r", encoding="utf-8") as f:
            subtitles = json.load(f)

    logging.info("字幕处理结束")

    logging.info("音频与字幕剪切开始")
    subtitles = deal_cut_video(path_manager, subtitles)
    with open(path_manager.subtitles_result_dir, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=4)
    logging.info("音频与字幕剪切完成")

    logging.info("音频分离人声开始")
    subtitles = deal_uvr_video(path_manager, subtitles)
    with open(path_manager.subtitles_result_dir, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=4)
    logging.info("音频分离人声完成")

    logging.info("ASR人声开始")
    subtitles = deal_asr(path_manager, subtitles)
    with open(path_manager.subtitles_result_dir, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=4)
    logging.info("ASR人声完成")

    logging.info("TTS人声开始")
    deal_tts(path_manager, subtitles)
    logging.info("TTS人声结束")


if __name__ == '__main__':
    main()

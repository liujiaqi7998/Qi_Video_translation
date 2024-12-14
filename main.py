import json
import logging
import os
import re

import pysubs2

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
TEMP_PATH = os.path.join(BASE_DIR, "TEMP")
sys.path.append(f"{BASE_DIR}")
sys.path.append(f"{BASE_DIR}/GPT_SoVITS")
import soundfile as sf
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names

# 这里可以配置参数


# 人声提取激进程度 0-20，默认10
agg = 10
input_language = "ja"
output_language = "zh"

# log_level = 'DEBUG'
log_level = 'INFO'

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=log_level)

cut_method_names = get_cut_method_names()
config_path = os.path.join(BASE_DIR, "GPT_SoVITS/configs/tts_infer.yaml")
tts_config = TTS_Config(config_path)
logging.debug(f"\n{tts_config}")
tts_pipeline = TTS(tts_config)

'''
临时目录设计
TEMP

input.mp4           输入视频
input.wav           输入视频提取原始音频
output.wav          输出音频
output.mp4          输出视频
subtitles.ass       视频字幕
- uvr5
      - instrument  背景音乐
      - vocal       视频人声
- cut 
      - instrument  背景音乐切片音频
      - vocal       视频人声切片音频
- translated
      - vocal       人声翻译音频音频
      - mix         背景音乐切片
      
'''

logging.info(f"程序启动成功，运行目录是：{BASE_DIR}")

# 先解决一下各功能模块目录问题

input_video_dir = os.path.join(TEMP_PATH, "input.mp4")
input_voice_dir = os.path.join(TEMP_PATH, "input.wav")
subtitles_dir = os.path.join(TEMP_PATH, "subtitles.ass")


output_video_dir = os.path.join(TEMP_PATH, "output.mp4")
output_voice_dir = os.path.join(TEMP_PATH, "output.wav")
output_voice_mp3_dir = os.path.join(TEMP_PATH, "output.mp3")

asr_result_dir = os.path.join(TEMP_PATH, "asr.json")
cut_result_dir = os.path.join(TEMP_PATH, "cut.json")
translate_result_dir = os.path.join(TEMP_PATH, "translate.json")
subtitles_result_dir = os.path.join(TEMP_PATH, "subtitles.json")

uvr5_instrument_dir = os.path.join(TEMP_PATH, "uvr5", "instrument")
uvr5_vocal_dir = os.path.join(TEMP_PATH, "uvr5", "vocal")

cut_instrument_dir = os.path.join(TEMP_PATH, "cut", "instrument")
cut_asr_vocal_dir = os.path.join(TEMP_PATH, "cut", "vocal_asr")

translated_vocal_dir = os.path.join(TEMP_PATH, "translated", "vocal")
translated_mix_dir = os.path.join(TEMP_PATH, "translated", "mix")

DirectoryOrCreate_dir = [
    uvr5_instrument_dir,
    uvr5_vocal_dir,
    cut_instrument_dir,
    cut_asr_vocal_dir,
    translated_vocal_dir,
    translated_mix_dir
]

# 检查一下目录是否存在，如果不存在就创建
for dir_path in DirectoryOrCreate_dir:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 第一步，利用ffmpeg将视频的音频提取出来
logging.info("ffmpeg，提取视频音频开始")

# Fix: 判断一下是否已经存在，存在就不再转换
if not os.path.exists(input_voice_dir):
    ffmpeg.input(input_video_dir).output(input_voice_dir, channels=2, sample_rate=44100).run(overwrite_output=True)

logging.info("ffmpeg，提取视频音频结束")

if not os.path.exists(input_voice_dir):
    raise FileNotFoundError(f"指定的音频文件不存在: {input_voice_dir}，可能是ffmpeg提取视频音频失败")


def detect_language_proportion(text):
    try:
        # 对输入文本进行语言检测，并返回每种语言的比例
        languages = detect_langs(text)
        # 将结果转换成字典格式方便查看和使用
        language_proportion = {str(lang.lang): lang.prob for lang in languages}
        return language_proportion
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {}


logging.info("字幕处理开始")

use_subs_style = {}

if not os.path.exists(subtitles_result_dir):
    subs = pysubs2.load(subtitles_dir)

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

    with open(subtitles_result_dir, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=4)
else:
    with open(subtitles_result_dir, "r", encoding="utf-8") as f:
        subtitles = json.load(f)

logging.info("使用uvr5提取人声开始")

instrument_input_wav = os.path.join(uvr5_instrument_dir, f"instrument_input.wav_{agg}.wav")
vocal_input_wav = os.path.join(uvr5_vocal_dir, f"vocal_input.wav_{agg}.wav")

if not (os.path.exists(instrument_input_wav) and os.path.exists(vocal_input_wav)):
    try:
        pre_fun = AudioPre(
            agg=agg,
            model_path=os.path.join(BASE_DIR, config.uvr5_weights_path, "HP5_only_main_vocal.pth"),
            device=config.infer_device,
            is_half=config.is_half,
        )
        pre_fun._path_audio_(
            music_file=input_voice_dir,
            ins_root=uvr5_instrument_dir,
            vocal_root=uvr5_vocal_dir,
            format="wav",
            is_hp3=False
        )
    except Exception as err:
        print(err)
    finally:
        try:
            del pre_fun.model
            del pre_fun
        except Exception as err:
            logging.error(f"uvr5异常：{err}")
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if not (os.path.exists(instrument_input_wav) and os.path.exists(vocal_input_wav)):
    raise FileNotFoundError(f"视频人声文件丢失，可能是uvr5失败")

logging.info("使用uvr5提取人声完成")

logging.info("开始提取人声内容")

if not os.path.exists(asr_result_dir):
    asr_model = whisper.load_model("large-v3",
                                   download_root=os.path.join(BASE_DIR, config.asr_models_path, "Whisper-large-v3"))
    asr_result = asr_model.transcribe(vocal_input_wav, language=input_language, initial_prompt=None)
    with open(asr_result_dir, "w", encoding="utf-8") as f:
        json.dump(asr_result, f, ensure_ascii=False, indent=4)
else:
    with open(asr_result_dir, "r", encoding="utf-8") as f:
        asr_result = json.load(f)

asr_segments = asr_result.get("segments")
logging.info(f"提取人声内容完成，共计{len(asr_segments)}条")

logging.info("开始分离提取人声")

change_list = []


if not os.path.exists(cut_result_dir):
    instrument_input_wav_audio = AudioSegment.from_mp3(file=instrument_input_wav)
    vocal_input_wav_audio = AudioSegment.from_mp3(file=vocal_input_wav)

    for asr_segment in asr_segments:
        asr_text = asr_segment.get("text")
        asr_id = asr_segment.get("id")
        asr_start = int(asr_segment.get("start") * 1000)
        asr_end = int(asr_segment.get("end") * 1000)
        count_time = asr_end - asr_start
        if count_time < 1500:
            # 小于1.5秒就拉倒吧，效果太次了
            logging.debug(f"跳过小于1.5秒的音频：{asr_id}")
            continue
        if len(asr_text) <= 2:
            logging.debug(f"跳过小于1字数：{asr_id}")
            continue

        asr_language = langid.classify(asr_text)
        if not asr_language:
            continue
        if not (input_language in asr_language[0]):
            logging.info(f"检测到的语言与目标语言不一致：{asr_text}")
            continue

        cut_instrument = instrument_input_wav_audio[asr_start:asr_end]
        cut_vocal = vocal_input_wav_audio[asr_start:asr_end]
        cut_vocal.export(os.path.join(cut_asr_vocal_dir, f"{asr_id}.wav"), format="wav")
        change_list.append(asr_segment)

    with open(cut_result_dir, "w", encoding="utf-8") as f:
        json.dump(change_list, f, ensure_ascii=False, indent=4)
else:
    with open(cut_result_dir, "r", encoding="utf-8") as f:
        change_list = json.load(f)

logging.info("人声提取完成")

logging.info("字幕与ARS数据匹配")

def find_segment_id(timestamp_sec):
    global change_list
    query_start = timestamp_sec - 0.1
    query_end = timestamp_sec + 0.1

    for segment in change_list:
        if query_start < segment["end"] and query_end > segment["start"]:
            return segment
    return None

for id , subtitle in subtitles.items():
    start = subtitle.get("start")
    if start == 0:
        continue
    asr_segment = find_segment_id(start/1000)
    if asr_segment:
        subtitles[id]["asr_segment"] = asr_segment

logging.info("字幕与ARS数据匹配完成")



logging.info("开始音频合成工作")

result_total = {
    "success": [],
    "failed": [],
    "out_of_time": []
}

for id,one in subtitles.items():
    a = "1"
    if asr_segment := one.get('asr_segment'):

        if os.path.exists(os.path.join(translated_vocal_dir, f'{id}.wav')):
            continue
            pass

        ref_audio_path = os.path.join(cut_asr_vocal_dir, f'{asr_segment.get("id")}.wav')
        if not os.path.exists(ref_audio_path):
            continue
            pass

        req = {
            "text": one.get("text"),  # str.(required) text to be synthesized
            "text_lang": output_language,  # str.(required) language of the text to be synthesized
            "ref_audio_path": ref_audio_path,  # str.(required) reference audio path
            "aux_ref_audio_paths": [],  # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
            "prompt_text": asr_segment.get("text"),  # str.(optional) prompt text for the reference audio
            "prompt_lang": input_language,  # str.(required) language of the prompt text for the reference audio
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

        try:
            output_wav = os.path.join(translated_vocal_dir, f'{id}.wav')
            tts_generator = tts_pipeline.run(req)
            sr, audio_data = next(tts_generator)
            sf.write(output_wav, audio_data, sr, format='wav')
            audio_segment = AudioSegment.from_wav(output_wav)
            duration_ms = len(audio_segment)
            cost_time = (one.get("end") - one.get("start"))
            if duration_ms > cost_time:
                speed = (duration_ms / cost_time) + 0.1
                logging.warning("输出音频时长大于原音频，需要压缩时长")
                new_speed_audio = audio_segment.speedup(playback_speed=speed)
                new_speed_audio.export(output_wav, format="wav")
                new_duration_ms = len(new_speed_audio)
                logging.debug(f"调整速度后的音频时长: {new_duration_ms} ms")
                result_total["out_of_time"].append(id)
            else:
                result_total["success"].append(id)
            pass
        except Exception as e:
            logging.error(f"TTS异常:{e}")
            result_total["failed"].append(id)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

logging.info(
    f"""音频合成完成：【成功】{len(result_total["success"])}个 |【失败】{len(result_total["failed"])}个 |【压缩时间】{len(result_total["out_of_time"])}个""")

logging.info("开始混音")
#
# for one in translate_list:
#     translated_wav = os.path.join(translated_vocal_dir, f'{one.get("id")}.wav')
#     instrument_wav = os.path.join(cut_instrument_dir, f'{one.get("id")}.wav')
#     output_wav = os.path.join(translated_mix_dir, f'{one.get("id")}.wav')
#     if os.path.exists(output_wav):
#         continue
#         pass
#     if not os.path.exists(translated_wav):
#         continue
#         pass
#     translated_audio = AudioSegment.from_file(translated_wav)
#     background_audio = AudioSegment.from_file(instrument_wav)
#     mixed_audio = background_audio.overlay(translated_audio)
#     mixed_audio.export(output_wav, format="wav")
#     pass
#
# logging.info("混音结束")
#
#
# logging.info("开始合并输出音频")
# input_voice_audio = AudioSegment.from_file(input_voice_dir)
# # combined_audio = AudioSegment.empty().set_channels(2)
#
#
#
# the_end_time = 0
# for i in range(len(translate_list)):
#     this_fragment = translate_list[i]
#     id = translate_list[i].get("id")
#     if not os.path.exists(os.path.join(translated_mix_dir, f'{id}.wav')):
#         continue
#     this_pic = AudioSegment.from_file(os.path.join(translated_mix_dir, f'{id}.wav'))
#     transl = this_fragment.get("transl")
#     start = this_fragment.get("start") * 1000
#     end = this_fragment.get("end") * 1000
#     input_voice_audio = input_voice_audio[:start] + this_pic + input_voice_audio[end:]
#     pass
#
# input_voice_audio.export(output_voice_dir, format='wav')
#
# logging.info("音频合成完成")
#
# input_voice_audio.export(output_voice_mp3_dir, format='mp3', bitrate="192k")
# logging.info("输出音频已压缩为MP3格式")
#
# exit(0)

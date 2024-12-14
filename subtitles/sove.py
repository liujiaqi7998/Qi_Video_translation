import json
import logging
import re

import pysubs2
from langdetect import detect_langs
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


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


output_language = "zh"
subs = pysubs2.load("/home/data/liujiaqi/Ai/GPT-SoVITS/TEMP/subtitles.ass")

use_subs_style = {}

for event in subs.events:
    if event.plaintext:
        asr_language = detect_language_proportion(event.plaintext)
        if not asr_language:
            continue
        if not (output_language in max(asr_language, key=asr_language.get)):
            logging.info(f"检测到的语言与目标语言不一致：{event.plaintext}")
            continue
        if not use_subs_style.get(event.style):
            use_subs_style[event.style] = 1
        else:
            use_subs_style[event.style] = use_subs_style[event.style] + 1
    pass

# 利用 max() 函数找到得分最高的项
most_use_style = max(use_subs_style, key=use_subs_style.get)

print(f"得分最高的style是: {most_use_style}，得分为: {use_subs_style[most_use_style]}")
# 已知问题：一个字幕会包含多种信息，如字幕制作人信息，多种语言混合，特效等待，但是他们的共同点就是采用了不同的style来区分，我们只需要获取到正确的
# style就可以解决一大半的问题了！
# 总体思路，已知字幕会使用不同的style作为分类，我们只需要通过判断语言，根据数量，选举出最有代表性的style，基本上就是有效字幕了（甜菜，甜菜，甜菜！！）

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


# 遍历每个 segment，检查它的 start 和 end 是否在查询区间内

with open("/home/data/liujiaqi/Ai/GPT-SoVITS/TEMP/asr.json", "r", encoding="utf-8") as f:
    asr_results = json.load(f)

def find_segment_id(timestamp_sec):
    global asr_results
    query_start = timestamp_sec - 0.1
    query_end = timestamp_sec + 0.1

    for segment in asr_results["segments"]:
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
    pass
a = "1"
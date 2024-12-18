import re
import langid
import pysubs2
from sqlalchemy.orm import sessionmaker
from config import output_language
from utils.db_utils import MainData
from loguru import logger


class SubtitlesExtraction:
    name = "字幕分割音频"
    session = None
    subtitles = []
    path_manager = None

    def __init__(self, engine, path_manager):
        if not path_manager:
            raise Exception("path_manager is None")
        if not engine:
            raise Exception("engine is None")
        self.session = sessionmaker(bind=engine)()
        self.path_manager = path_manager

    def close_session(self):
        if self.session:
            # 结束记得提交,数据才能保存在数据库中
            self.session.commit()
            # 关闭会话
            self.session.close()

    def log(self, msg, n_id=""):
        if n_id:
            n_id = f"|{n_id}|"
        logger.info(f"【{self.name}】{n_id}{msg}")

    def main(self):
        if not self.session:
            raise Exception("数据库未初始化")
        try:
            if self.session.query(MainData).first():
                self.log("数据库中存在数据，如需从新截取字幕请删除数据库然后重启程序")
                return

            use_subs_style = {}
            subs = pysubs2.load(self.path_manager.subtitles_dir)
            if not subs:
                raise Exception("字幕文件不生效")

            for event in subs.events:
                if event.plaintext:
                    language = langid.classify(event.plaintext)
                    if not language:
                        continue
                    if not (output_language in language[0]):
                        self.log(f"检测到的语言与目标语言不一致：{event.plaintext}")
                        continue
                    if not use_subs_style.get(event.style):
                        use_subs_style[event.style] = 1
                    else:
                        use_subs_style[event.style] = use_subs_style[event.style] + 1
                pass
            # 利用 max() 函数找到得分最高的项
            most_use_style = max(use_subs_style, key=use_subs_style.get)
            self.log(f"字幕得使用率最高的style是: {most_use_style}，得分为: {use_subs_style[most_use_style]}")
            start_id = 0
            for event in subs.events:
                if event.plaintext and event.style == most_use_style:
                    pattern = r"(www.|http|字幕组)"
                    if re.search(pattern, event.plaintext):
                        self.log(f"无意义字幕组: {event.plaintext}")
                        continue
                    # 解决一下字幕包含括号注解问题，如果发现这个字幕完全是个注释那么直接跳过处理
                    event.plaintext = re.sub(r'\(.*?\)', '', event.plaintext)
                    if len(event.plaintext) <= 0:
                        continue
                    self.subtitles.append(
                        MainData(id=start_id, start_time=event.start, end_time=event.end,
                                 subtitle_text=event.plaintext))
                    start_id = start_id + 1
        finally:
            self.session.add_all(self.subtitles)
            self.close_session()
            self.log(f"{self.name}结束")

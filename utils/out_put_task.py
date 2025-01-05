import os

from loguru import logger
from pydub import AudioSegment
from sqlalchemy.orm import sessionmaker

import config
from utils.db_utils import MainData


class OutPutTask:
    name = "输出混流"
    session = None
    path_manager = None

    def __init__(self, engine, path_manager):
        if not path_manager:
            raise Exception("path_manager is None")
        if not engine:
            raise Exception("engine is None")
        self.session = sessionmaker(bind=engine)()
        self.path_manager = path_manager
        logger.info(f"{self.name}开始")

    def close_session(self):
        if self.session:
            # 结束记得提交,数据才能保存在数据库中
            self.session.commit()
            # 关闭会话
            self.session.close()

    def log(self, msg, n_id="", level="INFO"):
        if n_id:
            n_id = f"|{n_id}|"
        logger.log(level, f"【{self.name}】{n_id}{msg}")

    def main(self):
        if not self.session:
            raise Exception("数据库未初始化")
        try:
            input_voice_audio = AudioSegment.from_file(self.path_manager.input_voice_dir)
            subtitles = self.session.query(MainData).all()
            for subtitle in subtitles:
                if not os.path.exists(os.path.join(self.path_manager.cut_mix_dir, f'{subtitle.id}.wav')):
                    continue
                this_pic = AudioSegment.from_file(os.path.join(self.path_manager.cut_mix_dir, f'{subtitle.id}.wav'))
                start = subtitle.start_time
                end = subtitle.end_time
                input_voice_audio = input_voice_audio[:start] + this_pic + input_voice_audio[end:]
            input_voice_audio.export(self.path_manager.output_voice_dir, format='wav')
            self.log("wav音频合成完成")
            input_voice_audio.export(self.path_manager.output_voice_mp3_dir, format='mp3', bitrate="192k")
            self.log("输出音频已压缩为MP3格式，方便传输测试")
            if config.combined_mkv:
                # 判断输入视频是否存在
                if not os.path.exists(self.path_manager.input_video_dir):
                    return
                self.log("开始合成视频")
                # 合成视频
                os.system(f"ffmpeg -i {self.path_manager.input_video_dir} -i {self.path_manager.output_voice_dir} -c copy -map 0 -map 1:a {self.path_manager.output_video_dir}")
                self.log("视频合成完成")
                
        finally:
            self.session.commit()
            self.close_session()
            self.log(f"{self.name}结束")

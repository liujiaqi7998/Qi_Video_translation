import os

from loguru import logger
from pydub import AudioSegment
from sqlalchemy.orm import sessionmaker

from utils.db_utils import MainData


class CutVideo:
    name = "视频切割"
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
        logger.info(f"{self.name}开始")

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
            subtitles = self.session.query(MainData).all()
            input_wav_audio = AudioSegment.from_file(file=self.path_manager.input_voice_dir)
            for subtitle in subtitles:
                output_path = os.path.join(self.path_manager.cut_asr_raw_dir, f"{subtitle.id}.wav")
                if not os.path.exists(output_path):
                    if subtitle.end_time - subtitle.start_time < 1000:
                        self.log(f"{subtitle.subtitle_text} 小于1秒，忽略采集", f"{subtitle.id}")
                        subtitle.cut_video_status = "小于1秒忽略"
                        continue
                    input_wav_audio[subtitle.start_time:subtitle.end_time].export(output_path, format='wav')
                subtitle.cut_video_status = "OK"
                pass
        finally:
            self.session.add_all(self.subtitles)
            self.close_session()
            self.log(f"{self.name}结束")

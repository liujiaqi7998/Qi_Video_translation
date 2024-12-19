import os
import shutil
import tempfile
import torch
from loguru import logger
from pydub import AudioSegment
from sqlalchemy.orm import sessionmaker
import config
from utils.db_utils import MainData


class MixVoice:
    name = "混合音频"
    session = None
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

    def log(self, msg, n_id="", level="INFO"):
        if n_id:
            n_id = f"|{n_id}|"
        logger.log(level, f"【{self.name}】{n_id}{msg}")

    def main(self):
        if not self.session:
            raise Exception("数据库未初始化")
        try:
            subtitles = self.session.query(MainData).all()
            for subtitle in subtitles:
                translated_wav = os.path.join(self.path_manager.cut_fix_dir, f'{subtitle.id}.wav')
                instrument_wav = os.path.join(self.path_manager.cut_instrument_dir, f'{subtitle.id}.wav')
                output_wav = os.path.join(self.path_manager.cut_mix_dir, f'{subtitle.id}.wav')

                if subtitle.mix_status == "OK":
                    # 如果生成过直接跳过
                    continue

                if (not os.path.exists(translated_wav)) or (not os.path.exists(instrument_wav)):
                    subtitle.mix_status = "跳过"
                    continue

                translated_audio = AudioSegment.from_file(translated_wav)
                background_audio = AudioSegment.from_file(instrument_wav)
                mixed_audio = background_audio.overlay(translated_audio)
                mixed_audio.export(output_wav, format="wav")
                subtitle.mix_status = "OK"
                self.session.commit()
        finally:
            self.session.commit()
            self.close_session()
            self.log(f"{self.name}结束")
            self.log("释放torch.cuda")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

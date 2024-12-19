import os
import shutil
import tempfile
import torch
from loguru import logger
from sqlalchemy.orm import sessionmaker
import config
from utils.db_utils import MainData


class OptimizationTask:
    name = "语音合成音频修复"
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
            with tempfile.TemporaryDirectory() as tmp:
                count = 0
                for subtitle in subtitles:
                    if subtitle.optimization_status == "OK":
                        continue

                    # if subtitle.tts_status != "OK":
                    #     subtitle.optimization_status = "跳过"
                    #     continue

                    # 这里进行人声处理
                    tts_path = os.path.join(self.path_manager.cut_tts_dir, f"{subtitle.id}.wav")
                    if not os.path.exists(tts_path):
                        subtitle.optimization_status = "TTS文件不存在"
                        continue

                    shutil.copyfile(tts_path, os.path.join(tmp, f"{subtitle.id}.wav"))
                    count = count + 1

                self.session.commit()

                self.log(f"一共{count}个音频需要提取人声")
                if count > 0:
                    os.system(f"{config.resemble_enhance_cmd} {tmp} {self.path_manager.cut_fix_dir}")
                self.log(f"语音修复完成")

                for subtitle in subtitles:
                    # 判断一下
                    fix_path = os.path.join(self.path_manager.cut_fix_dir, f"{subtitle.id}.wav")
                    if os.path.exists(fix_path):
                        subtitle.optimization_status = "OK"
        finally:
            self.session.commit()
            self.close_session()
            self.log(f"{self.name}结束")
            self.log("释放torch.cuda")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

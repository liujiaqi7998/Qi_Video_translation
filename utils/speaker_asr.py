import os

import torch
import whisper
from loguru import logger
from pydub import AudioSegment
from retrying import retry
from sqlalchemy.orm import sessionmaker

import config
from utils.db_utils import MainData


class SpeakerASR:
    name = "人声ASR识别"
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
            self.log("加载whisper的large-v3模型")
            asr_model = whisper.load_model("large-v3",
                                           download_root=str(os.path.join(config.BASE_DIR, config.asr_models_path,
                                                                          "Whisper-large-v3")))
            subtitles = self.session.query(MainData).all()
            for subtitle in subtitles:

                if subtitle.speaker_separation_status != "OK":
                    subtitle.asr_status = "跳过"
                    continue

                if subtitle.asr_status == "OK":
                    continue

                vocal_path = os.path.join(self.path_manager.cut_asr_vocal_dir, f"{subtitle.id}.wav")

                if subtitle.end_time - subtitle.start_time > 9000:
                    self.log(f"{subtitle.subtitle_text} 大于9秒，强制9秒截断", f"{subtitle.id}")
                    audio_segment = AudioSegment.from_wav(vocal_path)
                    vocal_path = os.path.join(self.path_manager.cut_asr_vocal_dir, f"{subtitle.id}_9s.wav")
                    audio_segment[subtitle.start_time:subtitle.start_time + 9000].export(vocal_path)
                    subtitle.asr_out_9s = 1

                if not os.path.exists(vocal_path):
                    continue

                @retry(stop_max_attempt_number=config.retry_times)
                def need_retry():
                    try:
                        asr_result = asr_model.transcribe(vocal_path, language=config.input_language,
                                                          initial_prompt=None)
                        if not (config.input_language in asr_result.get('language')):
                            self.log(
                                f"检测到{subtitle.id}的语言与目标语言不一致：{subtitle.subtitle_text}，结果：{asr_result.get('text')}",
                                f"{subtitle.id}")
                            return
                        subtitle.asr_text = asr_result.get("text")
                        subtitle.asr_language = asr_result.get("language")
                        self.log(f"识别：{subtitle.subtitle_text}，结果：{subtitle.asr_text}", f"{subtitle.id}")
                        subtitle.asr_extended = asr_result
                        subtitle.asr_status = "OK"
                        self.session.commit()
                    except Exception as err:
                        self.log(f"识别：{subtitle.subtitle_text}，发生异常：{err}，触发重试", f"{subtitle.id}", "WARNING")
                        raise err
                try:
                    need_retry()
                except Exception as err:
                    subtitle.asr_status = str(err)
                    self.log(f"识别：{subtitle.subtitle_text}，重试失败完全异常:{err}", f"{subtitle.id}", "ERROR")
                    continue

        finally:
            self.session.commit()
            self.close_session()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.log(f"{self.name}结束")

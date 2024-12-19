import os
import sys
import torch
import soundfile as sf
from loguru import logger
from pydub import AudioSegment
from retrying import retry
from sqlalchemy.orm import sessionmaker
import config
from utils.db_utils import MainData


class TTSTask:
    name = "语音合成"
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
            sys.path.append(f"{config.BASE_DIR}")
            sys.path.append(f"{config.BASE_DIR}/GPT_SoVITS")
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
            config_path = os.path.join(config.BASE_DIR, "GPT_SoVITS/configs/tts_infer.yaml")
            tts_config = TTS_Config(config_path)
            logger.debug(f"\n{tts_config}")
            tts_pipeline = TTS(tts_config)

            subtitles = self.session.query(MainData).all()
            for subtitle in subtitles:
                if subtitle.tts_status == "OK":
                    continue

                if subtitle.asr_status != "OK":
                    subtitle.tts_status = "跳过"
                    continue

                ref_audio_path = os.path.join(self.path_manager.cut_asr_vocal_dir, f"{subtitle.id}.wav")

                if subtitle.asr_out_9s == 1:
                    # 超过9秒，使用9秒裁剪音频
                    ref_audio_path = os.path.join(self.path_manager.cut_asr_vocal_dir, f"{subtitle.id}_9s.wav")

                if not os.path.exists(ref_audio_path):
                    # 解决 ref_audio_path 音频不存在问题
                    subtitle.tts_status = "语音合成参考音频不存在"
                    continue

                req = {
                    "text": subtitle.subtitle_text,  # str.(required) text to be synthesized
                    "text_lang": config.output_language,  # str.(required) language of the text to be synthesized
                    "ref_audio_path": ref_audio_path,  # str.(required) reference audio path
                    "aux_ref_audio_paths": [],
                    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                    "prompt_text": subtitle.asr_text,  # str.(optional) prompt text for the reference audio
                    "prompt_lang": subtitle.asr_language,
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

                @retry(stop_max_attempt_number=config.retry_times)
                def need_retry():
                    try:
                        output_wav = os.path.join(self.path_manager.cut_tts_dir, f'{subtitle.id}.wav')
                        if os.path.exists(output_wav):
                            audio_segment = AudioSegment.from_wav(output_wav)
                            duration_ms = len(audio_segment)
                            cost_time = (subtitle.end_time - subtitle.start_time)
                            if duration_ms > cost_time:
                                # 判断出来确实有必要修改一下配音速度，否则就直接跳过这个
                                speed = (duration_ms / cost_time) + 0.1
                                self.log(f"检测到音频存在，压缩时长到{speed}")
                                req["speed_factor"] = speed
                            else:
                                return
                        tts_generator = tts_pipeline.run(req)
                        sr, audio_data = next(tts_generator)
                        sf.write(output_wav, audio_data, sr, format='wav')
                        audio_segment = AudioSegment.from_wav(output_wav)
                        duration_ms = len(audio_segment)
                        cost_time = (subtitle.end_time - subtitle.start_time)
                        if duration_ms > cost_time:
                            subtitle.tts_status = "时长超限"
                            raise Exception("输出音频时长大于原音频，需要压缩时长")
                        subtitle.tts_status = "OK"
                        self.session.commit()
                    except Exception as e:
                        self.log(f"处理{subtitle.subtitle_text} 发生异常:{e}，触发重试", str(subtitle.id), "WARNING")
                        raise e

                try:
                    need_retry()
                except Exception as err:
                    self.log(f"处理{subtitle.subtitle_text} 发生异常:{err}", str(subtitle.id), "ERROR")
        finally:
            self.session.commit()
            self.close_session()
            self.log(f"{self.name}结束")
            logger.info("释放torch.cuda")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

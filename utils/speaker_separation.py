import os
import shutil
import tempfile
import torch
from loguru import logger
from pydub import AudioSegment
from sqlalchemy.orm import sessionmaker

import config
from tools.uvr5.vr import AudioPre
from utils.db_utils import MainData
from utils.file_path import PathManager


class SpeakerSeparation:
    name = "视频提取人声"
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
            if not os.path.exists(self.path_manager.instrument_dir):
                raise Exception("没有有效的背景音频文件，可能是上一步人声分离失败")
            instrument_audio = AudioSegment.from_file(self.path_manager.instrument_dir)

            # 这里进行人声处理
            with tempfile.TemporaryDirectory() as tmp:
                count = 0
                subtitles = self.session.query(MainData).all()
                for subtitle in subtitles:
                    raw_path = os.path.join(self.path_manager.cut_asr_raw_dir, f"{subtitle.id}.wav")
                    if subtitle.cut_video_status != "OK":
                        subtitle.speaker_separation_status = "跳过"
                        continue
                    if subtitle.speaker_separation_status == "OK":
                        # 发现已经处理过了直接跳过
                        continue
                    if not os.path.exists(raw_path):
                        subtitle.speaker_separation_status = "音频片段文件不存在"
                        continue
                    shutil.copyfile(raw_path, os.path.join(tmp, f"{subtitle.id}.wav"))
                    count = count + 1

                self.session.commit()
                self.log(f"一共{count}个音频需要提取人声")

                if count > 0:
                    os.system(f"{config.resemble_enhance_cmd} {tmp} {self.path_manager.cut_asr_vocal_dir}")
                self.log(f"人声提取完成")
                # 背景音频裁切
                subtitles = self.session.query(MainData).all()
                for subtitle in subtitles:
                    instrument_path = os.path.join(self.path_manager.cut_instrument_dir, f"{subtitle.id}.wav")
                    if os.path.exists(instrument_path):
                        continue
                    instrument_audio[subtitle.start_time:subtitle.end_time].export(instrument_path)
                    if os.path.exists(instrument_path):
                        subtitle.speaker_separation_status = "OK"
        finally:
            self.session.commit()
            self.close_session()
            self.log(f"{self.name}结束")


def deal_uvr_all_video(path_manager: PathManager):
    pre_fun = None

    if os.path.exists(path_manager.instrument_dir) and os.path.exists(path_manager.vocal_dir):
        logger.info(f"视频提取背景音频已经存在，跳过次步骤，如下重新生成请删除{path_manager.instrument_dir}")
        return

    try:
        logger.info("开始视频提取背景音频")

        pre_fun = AudioPre(
            agg=config.agg,
            model_path=os.path.join(config.BASE_DIR, config.uvr5_weights_path, "HP5_only_main_vocal.pth"),
            device=config.infer_device,
            is_half=config.is_half,
        )

        audio = AudioSegment.from_file(path_manager.input_voice_dir)
        with tempfile.TemporaryDirectory() as tmp:
            audio_mono_list = audio.split_to_mono()
            j = 0
            for one_mono in audio_mono_list:
                music_file = os.path.join(tmp, f"{j}.wav")
                ins_path_file = os.path.join(tmp, f"{j}_ins.wav")
                vocal_path_file = os.path.join(tmp, f"{j}_vocal.wav")
                logger.info(f"【{j}声道】开始导出数据，请等待")
                one_mono.export(music_file)
                if not os.path.exists(music_file):
                    raise Exception(f"【{j}声道】导出异常，文件没有成功写出")

                logger.info(f"【{j}声道】全视频人声分离开始，这个过程需要数分钟，请耐心等待")
                pre_fun._path_audio_(
                    music_file=music_file,
                    ins_path=ins_path_file,
                    vocal_path=vocal_path_file,
                )
                if (not os.path.exists(ins_path_file)) or (not os.path.exists(vocal_path_file)):
                    raise Exception(f"【{j}声道】分离失败，文件没有成功写出")
                j = j + 1

            u_ins_audio = []
            u_vocal_audio = []
            for m in range(j):
                ins_path_file = os.path.join(tmp, f"{m}_ins.wav")
                vocal_path_file = os.path.join(tmp, f"{m}_vocal.wav")
                l_ins_audio, r_ins_audio = AudioSegment.from_file(ins_path_file).split_to_mono()
                if l_ins_audio.rms < r_ins_audio.rms:
                    u_ins_audio.append(l_ins_audio)
                else:
                    u_ins_audio.append(r_ins_audio)
                l_vocal_audio, r_vocal_audio = AudioSegment.from_file(vocal_path_file).split_to_mono()
                # 防止爆音 选择能量低的一组
                if l_vocal_audio.rms < r_vocal_audio.rms:
                    u_vocal_audio.append(l_vocal_audio)
                else:
                    u_vocal_audio.append(r_vocal_audio)

            AudioSegment.from_mono_audiosegments(*u_ins_audio).export(path_manager.instrument_dir)
            AudioSegment.from_mono_audiosegments(*u_vocal_audio).export(path_manager.vocal_dir)
    except Exception as err:
        logger.error(err)
    finally:
        logger.info("视频提取背景音频完成")
        try:
            if pre_fun:
                del pre_fun.model
                del pre_fun
        except Exception as err:
            logger.error(f"uvr5异常：{err}")
        logger.info("释放torch.cuda")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return

from sqlalchemy import Column, Integer, String, create_engine, Date, TIMESTAMP, func, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class MainData(Base):
    __tablename__ = 'main'
    id = Column(Integer, primary_key=True, autoincrement=True, comment="ID")

    start_time = Column(Integer, comment="字幕开始时间")
    end_time = Column(Integer, comment="字幕结束时间")
    subtitle_text = Column(String, nullable=True, comment="字幕")
    asr_text = Column(String, nullable=True, comment="ASR 识别的内容")
    asr_language = Column(String, nullable=True, comment="ASR 识别出来的语音")
    asr_extended = Column(JSON, nullable=True, comment="ASR 原始JSON内容")
    asr_out_9s = Column(Integer, default=0, comment="是否超过9秒")

    # 下面是进度追踪用
    cut_video_status = Column(String, nullable=True, comment="音频根据字幕分片")
    speaker_separation_status = Column(String, nullable=True, comment="人声提取")
    asr_status = Column(String, nullable=True, comment="语音识别")
    tts_status = Column(String, nullable=True, comment="语音合成")
    optimization_status = Column(String, nullable=True, comment="语音合成优化")
    mix_status = Column(String, nullable=True, comment="混音")

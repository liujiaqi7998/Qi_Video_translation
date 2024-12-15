import os


class PathManager:
    '''
    临时目录设计
    TEMP

    input.mp4           输入视频
    input.wav           输入视频提取原始音频
    output.wav          输出音频
    output.mp4          输出视频
    subtitles.ass       视频字幕
    - cache             缓存
    - uvr5
          - instrument  背景音乐
          - vocal       视频人声
    - cut
          - instrument  背景音乐切片音频
          - vocal       视频人声切片音频
    - translated
          - vocal       人声翻译音频音频
          - mix         背景音乐切片
    - speakers
          - 1.wav       讲述人
    '''


    def __init__(self, base_path):
        self.base_path = base_path

        # Initialize all paths
        self.input_video_dir = os.path.join(base_path, "input.mp4")
        self.input_voice_dir = os.path.join(base_path, "input.wav")
        self.subtitles_dir = os.path.join(base_path, "subtitles.ass")

        self.output_video_dir = os.path.join(base_path, "output.mp4")
        self.output_voice_dir = os.path.join(base_path, "output.wav")
        self.output_voice_mp3_dir = os.path.join(base_path, "output.mp3")

        self.asr_result_dir = os.path.join(base_path, "asr.json")
        self.cut_result_dir = os.path.join(base_path, "cut.json")
        self.translate_result_dir = os.path.join(base_path, "translate.json")
        self.subtitles_result_dir = os.path.join(base_path, "subtitles.json")
        self.pyannote_result_dir = os.path.join(base_path, "pyannote.json")
        self.speaker_result_dir = os.path.join(base_path, "speaker.json")

        self.uvr5_instrument_dir = os.path.join(base_path, "uvr5", "instrument")
        self.uvr5_vocal_dir = os.path.join(base_path, "uvr5", "vocal")

        self.speakers_dir = os.path.join(base_path, "speakers")
        self.cache_dir = os.path.join(base_path, "cache")
        self.cut_instrument_dir = os.path.join(base_path, "cut", "instrument")
        self.cut_asr_vocal_dir = os.path.join(base_path, "cut", "vocal_asr")
        self.cut_asr_raw_dir = os.path.join(base_path, "cut", "raw")

        self.translated_vocal_dir = os.path.join(base_path, "translated", "vocal")
        self.translated_mix_dir = os.path.join(base_path, "translated", "mix")

        self.directories_to_create = [
            self.uvr5_instrument_dir,
            self.uvr5_vocal_dir,
            self.cut_instrument_dir,
            self.cut_asr_vocal_dir,
            self.cut_asr_raw_dir,
            self.translated_vocal_dir,
            self.translated_mix_dir,
            self.speakers_dir,
            self.cache_dir
        ]

    def create_directories(self):
        # Check and create directories if they don't exist
        for dir_path in self.directories_to_create:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
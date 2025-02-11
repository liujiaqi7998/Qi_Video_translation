# 奇慧智译

项目施工中

一款基于AI实现的自动化视频翻译工具


# 如何使用

很抱歉我们仅在Linux x64(ubuntu) 上测试开发，您如果使用其他平台，请自行测试兼容性

### 环境准备

克隆项目到本地

```bash
git clone https://github.com/liujiaqi7998/Qi_Video_translation.git
```

安装环境

```bash
conda create -n Qi_Video_translation python=3.9
conda activate Qi_Video_translation
conda install -c conda-forge gcc
conda install -c conda-forge gxx
conda install ffmpeg cmake
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

准备模型

建议从 https://github.com/RVC-Boss/GPT-SoVITS 文档拉取模型


### 文件准备

1. 创建文件夹 `TEMP`
2. 将视频放入文件夹中，重命名为 `input.mp4`
3. 如果文件不是视频格式，请将音频文件放入文件夹，重命名为 `input.wav`
4. 将字幕文件放入文件夹，重命名为  `subtitles.ass`
5. 字幕文件的质量影响合成的效果，请保证字幕的时间轴与视频对齐，请先完成这个步骤。


### 开始使用

执行下面的命令开始处理

```bash
python main.py -path "TEMP"
```

多显卡用户请在启动前通过 `CUDA_VISIBLE_DEVICES=1` 环境变量设置要使用的显卡

### 取出文件

生成的视频文件在 `./TEMP/output.mkv`

生成的音频文件在 `./TEMP/output.wav`

为了方便预览会生成 `./TEMP/output.mp3` 文件，小体积有损压缩便于在节约网络带宽的场景下使用

### 使用容器

使用Dockerfile构架

### 使用webUI（Beta）

webUI 使用的是 gradio 库，可以方便的进行任务调度，该功能还在开发中，请耐心等待

```bash
python webui.py
```


# 特别鸣谢

我们使用了绝大多数的代码来自：

https://github.com/RVC-Boss/GPT-SoVITS

特别感谢以下项目和贡献者：

https://github.com/resemble-ai/resemble-enhance

https://huggingface.co/lj1995/VoiceConversionWebUI

https://github.com/RubyMetric/chsrc

https://github.com/FFmpeg/FFmpeg

https://github.com/alibaba-damo-academy/FunASR

https://github.com/SYSTRAN/faster-whisper

https://github.com/gradio-app/gradio

https://github.com/jaywalnut310/vits

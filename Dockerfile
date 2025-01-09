FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

LABEL maintainer="Liu Jiaqi"

# 禁用交互
ENV DEBIAN_FRONTEND=noninteractive

# 设置时区
ENV TZ=Asia/Shanghai

# 安装 chsrc 换源
RUN apt update && apt install curl -y
COPY scripts/chsrc /bin/chsrc
RUN chsrc set ubuntu

# 上质量，先解决一下Ubuntu不完整的问题
RUN apt update && \
    apt install -y wget sudo git curl vim net-tools htop wget gcc ffmpeg iputils-ping tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    yes | unminimize

# 设置一下用户目录
WORKDIR /root

# 下载安装conda环境
RUN  wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3.sh && \
        /bin/bash ~/miniconda3.sh -b -p ~/miniconda3 && \
        rm ~/miniconda3.sh && \
        chsrc set conda && \
        echo ". /root/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
        /root/miniconda3/bin/conda init bash &&\
        chsrc set conda



RUN /root/miniconda3/bin/conda create -n resemble-enhance python=3.10

RUN export PATH="/root/miniconda3/envs/resemble-enhance/bin/:$PATH" && chsrc set python

RUN /root/miniconda3/envs/resemble-enhance/bin/pip install resemble-enhance --upgrade

RUN /root/miniconda3/bin/conda create -n Qi_Video_translation python=3.9

RUN /root/miniconda3/bin/conda install -n Qi_Video_translation pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
RUN apt install cmake gcc g++ -y
COPY requirements.txt /tmp/requirements.txt
RUN /root/miniconda3/envs/Qi_Video_translation/bin/pip install -r /tmp/requirements.txt

# 拷贝到 app
COPY . /app

RUN cp -r 

# 设置工作目录
WORKDIR /app

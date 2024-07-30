From nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 设置环境变量
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138 \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0E98404D386FA1D9 \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys DCC9EFBF77E11517 \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 112695A0E562B32A \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 54404762BBB6E853

# 更新并安装软件包
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

RUN apt-get update && \
    apt-get install -y libc6-dev g++ && \
    apt-get clean

# 下载并安装Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda -u && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -t -i -p -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean --all -y

WORKDIR /usr/src/app

RUN echo 'channels:\n\
  - defaults\n\
show_channel_urls: true\n\
default_channels:\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\n\
custom_channels:\n\
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud'\
> /root/.condarc

RUN echo 'deb http://mirrors.tuna.tsinghua.edu.cn/debian/ buster main contrib non-free\n\
deb http://mirrors.tuna.tsinghua.edu.cn/debian/ buster-updates main contrib non-free\n\
deb http://mirrors.tuna.tsinghua.edu.cn/debian/ buster-backports main contrib non-free\n\
deb http://mirrors.tuna.tsinghua.edu.cn/debian-security buster/updates main contrib non-free'\
> /etc/apt/sources.list


RUN apt-get --allow-releaseinfo-change update && apt-get update -y \ 
 && apt-get install -y libc6-dev g++ \ 
 && apt-get install -y build-essential libgl1-mesa-glx \ 
 && rm -rf /var/lib/apt/lists/*


COPY environment.yaml ./

RUN  conda env create -f ./environment.yaml && conda clean --tarballs -y

# # 设置CUDA_HOME环境变量
ENV CUDA_HOME /usr/local/cuda


ARG TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
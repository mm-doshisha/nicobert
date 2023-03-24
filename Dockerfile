FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# Install linux packages
RUN apt update \
&& apt upgrade -y \
&& apt install -y python3 python3-pip libgl1-mesa-glx sudo git curl mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8 unidic-mecab

RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git /tmp/mecab-ipadic-neologd \
&& /tmp/mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y \
&& rm -rf /tmp/mecab-ipadic-neologd
RUN ln -s /etc/mecabrc /usr/local/etc/

# settings for japanese
RUN apt install -y locales \
&& locale-gen ja_JP.UTF-8 \
&& echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

# Install python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip \
&& pip --no-cache-dir install -r requirements.txt
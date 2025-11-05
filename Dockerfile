FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app

SHELL ["/bin/bash", "-c"]
ENV HF_HOME=/huggingface

RUN apt update && apt install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN apt update && apt install -y --no-install-recommends gnupg
RUN echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt update && apt install -y nsight-systems-cli
RUN uv venv /.venv --python 3.10 --seed && source /.venv/bin/activate && uv pip install vllm==0.6.5 matplotlib datasets==4.0.0
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# [Optional] Uncomment this section to install additional OS packages.
RUN apt update && export DEBIAN_FRONTEND=noninteractive \
    && apt install -y --no-install-recommends \
    cmake \
    libcunit1 libcunit1-dev \
    valgrind \ 
    opencl-headers ocl-icd-opencl-dev \
    pocl-opencl-icd \
    clinfo 

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-8

USER vscode

RUN echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc

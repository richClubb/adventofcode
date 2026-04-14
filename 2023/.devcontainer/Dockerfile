FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# [Optional] Uncomment this section to install additional OS packages.
RUN apt update && export DEBIAN_FRONTEND=noninteractive \
    && apt install -y --no-install-recommends \
    cmake \
    libcunit1 libcunit1-dev \
    valgrind \ 
    opencl-headers ocl-icd-opencl-dev \
    pocl-opencl-icd \
    maven \
    clinfo \
    clang

RUN wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb && \
    dpkg -i cuda-repo-debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb && \
    cp /var/cuda-repo-debian12-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-8

USER vscode

RUN echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc

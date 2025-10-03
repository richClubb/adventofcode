#FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04
FROM mcr.microsoft.com/devcontainers/base:bookworm

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    cmake \
    libcunit1 libcunit1-dev \
    valgrind

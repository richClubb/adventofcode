FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# [Optional] Uncomment this section to install additional OS packages.
RUN apt update && export DEBIAN_FRONTEND=noninteractive \
    && apt install -y --no-install-recommends \
    clinfo 

USER vscode


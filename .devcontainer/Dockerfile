ARG PYTHON_VERSION
FROM mcr.microsoft.com/vscode/devcontainers/python:${PYTHON_VERSION}

RUN apt update && apt install -y locales git clang-format ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

ARG TF_VERSION
ARG KERASCV_VERSION
RUN pip install \
    tensorflow-cpu==${TF_VERSION} \
    keras-cv==${KERASCV_VERSION} \
    # demo
    matplotlib \
    opencv-python-headless \
    # pytest
    ruff \
    black[jupyter] \
    pycocotools \
    pytest \
    pytest-cov \
    # docs
    sphinx~=6.1.3 \
    sphinx_markdown_tables~=0.0.17 \
    sphinx_rtd_theme~=1.2.0 \
    myst-parser~=1.0.0 \
    # dist
    twine

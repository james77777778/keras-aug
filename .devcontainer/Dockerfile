ARG PYTHON_VERSION
ARG TF_VERSION
ARG TFP_VERSION
FROM mcr.microsoft.com/vscode/devcontainers/python:${PYTHON_VERSION}

RUN apt update && apt install -y locales git clang-format \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install \
    tensorflow-cpu==${TF_VERSION} \
    tensorflow_probability==${TFP_VERSION} \
    git+https://github.com/keras-team/keras-cv.git \
    matplotlib \
    flake8 \
    isort \
    black[jupyter] \
    pycocotools \
    pytest \
    pytest-cov \
    # docs
    sphinx~=6.1.3 \
    sphinx_markdown_tables~=0.0.17 \
    sphinx_rtd_theme~=1.2.0 \
    # dist
    twine

FROM mcr.microsoft.com/vscode/devcontainers/python:3.8
RUN mkdir requirements
COPY requirements/docs.txt requirements/docs.txt
COPY requirements/dist.txt requirements/dist.txt
COPY setup.sh /setup.sh
RUN sh setup.sh

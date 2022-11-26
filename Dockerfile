FROM ucsdets/datahub-base-notebook:2022.3-stable

LABEL maintainer="Camille"

WORKDIR /

COPY src src
COPY config config
COPY data/test data/test

COPY run.py run.py
COPY setup.py setup.py

RUN conda install pytorch==1.12.1 -c pytorch
RUN conda install pyg -c pyg
RUN pip install --no-cache-dir \
    typing_extensions==4.4.0 \
    wandb==0.13.5 \
    pyyaml~=6.0 \
    pytorch_lightning==1.8.3.post0 \
    numpy==1.23.5

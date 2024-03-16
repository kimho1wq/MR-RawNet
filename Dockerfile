FROM nvcr.io/nvidia/pytorch:23.07-py3

RUN apt-get update
RUN pip install pip --upgrade

RUN pip install torch==2.0.1
RUN pip install torchaudio==2.0.1

ENV PYTHONPATH /workspace/MR-RawNet
WORKDIR /workspace/MR-RawNet

RUN apt-get install git-lfs
RUN pip install wandb --upgrade
RUN pip install neptune

RUN pip install h5py
RUN pip install yamlargparse
RUN pip install soundfile

RUN pip install asteroid_filterbanks==0.4.0

# MR-RawNet

This repository contains official pytorch implementation and pre-trained models for following paper:
  - Title : MR-RawNet: Speaker verification system with multiple temporal resolutions for variable duration utterances using raw waveforms
  - Autor : Seung-bin Kim, Chan-yeong Lim, Jungwoo Heo, Ju-ho Kim, Hyun-seo Shin, Kyo-Won Koo and Ha-Jin Yu

# Abstract
![overall](https://github.com/kimho1wq/TIL/assets/15611500/d9bbd1b9-da57-435d-aebf-2ee148912cd3)

In speaker verification systems, the utilization of short utterances presents a persistent challenge, leading to performance degradation primarily due to insufficient phonetic information to characterize the speakers.
To overcome this obstacle, we propose a novel structure, MR-RawNet, designed to enhance the robustness of speaker verification systems against variable duration utterances using raw waveforms. 
The MR-RawNet extracts time-frequency representations from raw waveforms via a multi-resolution feature extractor that optimally adjusts both temporal and spectral resolutions simultaneously.
Furthermore, we apply a multi-resolution attention block that focuses on diverse and extensive temporal contexts, ensuring robustness against changes in utterance length.
The experimental results, conducted on VoxCeleb1 dataset, demonstrate that the MR-RawNet exhibits superior performance in handling utterances of variable duration compared to other raw waveform-based systems.


Our experimental code was modified based on [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer), and we referenced the baseline code at [here](https://github.com/Jungjee/RawNet).


# Data
The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets were used for training and test.

The train list should contain the identity and the file path, one line per utterance, as follows:
```
id00000 id00000/youtube_key/12345.wav
id00012 id00012/21Uxsk56VDQ/00001.wav
```
The train list for VoxCeleb2 can be download from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt), and the test lists for VoxCeleb1 can be downloaded from [here](https://mm.kaist.ac.kr/datasets/voxceleb/index.html#testlist). 


For data augmentation, the following script can be used to download and prepare.
```
python3 ./dataprep.py --save_path data --augment
```

We also performed an out-of-domain evaluation using the [VOiCES](https://iqtlabs.github.io/voices/downloads/) development set.

Each dataset must be downloaded in advance for training and testing, and its path must be mapped to the docker environment.

# Environment
Docker image (nvcr.io/nvidia/pytorch:23.07-py3) of Nvidia GPU Cloud was used for conducting our experiments.

Make docker image and activate docker container.
```
./docker/build.sh
./docker/run.sh
```

Note that you need to modify the mapping path before running the 'run.sh' file.

# Training

- MR-RawNet on a single GPU
```
python3 ./trainSpeakerNet.py --config ./configs/MR_RawNet.yaml
```

- MR-RawNet on multiple GPUs
```
CUDA_VISIBLE_DEVICES=0,1 python3 ./trainSpeakerNet.py --config ./configs/MR_RawNet.yaml --distributed
```
Use --distributed flag to enable distributed training.
If you are running more than one distributed training session, you need to change the --port argument.

Note that the configuration file overrides the arguments passed via command line.

# Test

The following script should return: `EER 0.8294`.
```
python3 ./trainSpeakerNet.py --eval --config ./configs/MR_RawNet.yaml --initial_model MR_RawNet.pt
```


# Citation
Please cite if you make use of the code.

```
@article{kim2024mrrawnet,
  title={MR-RawNet: Speaker verification system with multiple temporal resolutions for variable duration utterances using raw waveforms},
  author={Kim, Seung-bin and Lim, Chan-yeong and Heo, Jungwoo and Kim, Ju-ho and Shin, Hyun-seo and Koo, Kyo-Won and Yu, Ha-Jin},
  journal={arXiv preprint arXiv:},
  year={2024}
}

@inproceedings{kim2024mrrawnet,
  title={MR-RawNet: Speaker verification system with multiple temporal resolutions for variable duration utterances using raw waveforms},
  author={Kim, Seung-bin and Lim, Chan-yeong and Heo, Jungwoo and Kim, Ju-ho and Shin, Hyun-seo and Koo, Kyo-Won and Yu, Ha-Jin},
  booktitle={Proc. Interspeech},
  year={2024}
}
```
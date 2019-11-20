# Source Separation

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAppleholic%2Fsource_separation)](https://hits.seeyoufarm.com)

---


## Introduction

*Source Separation* is a repository to extract speeches from various recorded sounds.
It focuses to adapt more real-like dataset for training models.

### Main components, different things

The latest model in this repository is basically built with spectrogram based models.
In mainly, [Phase-aware Speech Enhancement with Deep Complex U-Net](https://arxiv.org/abs/1903.03107) are implemented with modifications.
- Complex Convolution, Masking, Weighted SDR Loss

And then, To more stable inferences in real cases, below things are adopted.

- Preemphasis is used to remove high-frequency noises on adapting real samples.

It's not official implementation by authors of paper.


### Recent Updates

#### Notice deprecation and report issue about voicebank pretrained model with audioset

##### It's under review in detail. I'm checking out codes for reproducing experiments, after then, below comments can be fixed.

After releasing public repo, I found a bug on augmentation on that trained model that did not trained with audioset.
Then, I recently did retry to train model and get slightly different result.

- (Checking) Initial samples and model are voicebank only version.


#### Singing Voice Separation

*Now, add Singing Voice Separation with [DSD100](https://sigsep.github.io/datasets/dsd100.html) dataset!*
This model is trained with larger model and higher sample rate (44.1k). So it gives more stable and high quality audio.
Let's checkout [Youtube Playlist](https://www.youtube.com/playlist?list=PLQ4ukFz6Ieir5bZYOns08_2gMjt4hYP4I) with samples of my favorites!


### Dataset

You can use pre-defined preprocessing and dataset sources on https://github.com/Appleholic/pytorch_sound


## List to be updated

- [ ] Add MUSDB and evaluate results (issue-9)
- [ ] Enhance codes for inference


## Environment

- Python > 3.6
- pytorch 1.0
- ubuntu 16.04
- Brain Cloud (Kakaobrain Cluster) V2.XLARGE (2 V100 GPUs, 28 cores cpu, 244 GB memory)


## External Repositories

They are two external repositories on this repository.
*These will be updated to setup with recursive clone or internal codes*

- pytorch_sound package

It is built with using [pytorch_sound](https://github.com/AppleHolic/pytorch_sound).
So that, *pytorch_sound* is a modeling toolkit that allows engineers to train custom models for sound related tasks.
Many of sources in this repository are based on pytorch_sound template.


## Pretrained Checkpoint

- General Voice Source Separation
  - Model Name : refine_unet_base (see settings.py)
  - Link : [Google Drive](https://drive.google.com/open?id=1JRK-0RVV2o7cyRdvFuwe5iw84ESvfcyR)
  - Available Tag : v0.1.0

- Singing Voice Separation
  - Model Name : refine_unet_larger
  - Link : [Google Drive](https://drive.google.com/open?id=1ywgFZ7ms7CmiCCv2MikrKx9g-2j9kd-I)
  - Available Tag : v0.1.0


## Predicted Samples

- *General Voice Source Separation*
  - Validation 10 random samples
    - Link : [Google Drive](https://drive.google.com/open?id=1CafFnqWn_QvVPu2feNLn6pnjRYIa_rbP)

  - Test Samples :
    - Link : [Google Drive](https://drive.google.com/open?id=19Sn6pe5-BtWXYa6OiLbYGH7iCU-mzB8j)

- *Singing Voice Seperation*
  - Check out my youtube playlist !
    - Link : [Youtube Playlist](https://www.youtube.com/playlist?list=PLQ4ukFz6Ieir5bZYOns08_2gMjt4hYP4I)


## Installation

- Install above external repos

> You should see first README.md of pytorch_sound, to prepare dataset and to train separation models.

```
$ pip install git+https://github.com/Appleholic/pytorch_sound@v0.0.3
```

- Install package

```bash
$ pip install -e .
```

## Usage

- Train

```bash
$ python source_separation/train.py [YOUR_META_DIR] [SAVE_DIR] [MODEL NAME, see settings.py] [SAVE_PREFIX] [[OTHER OPTIONS...]]
```

- Joint Train (Voice Bank and DSD100)

```bash
$ python source_separation/train_jointly.py [YOUR_VOICE_BANK_META_DIR] [YOUR_DSD100_META_DIR] [SAVE_DIR] [MODEL NAME, see settings.py] [SAVE_PREFIX] [[OTHER OPTIONS...]]
```


- Synthesize
  - Be careful the differences sample rate between general case and singing voice case!
  - If you run more than one, it can help to get better result.
    - Sapmles of singing voice separation are ran twice.

Single sample

```bash
$ python source_separation/synthesize.py separate [INPUT_PATH] [OUTPUT_PATH] [MODEL NAME] [PRETRAINED_PATH] [[OTHER OPTIONS...]]
```


Whole validation samples

```bash
$ python source_separation/synthesize.py validate [YOUR_META_DIR] [OUTPUT_DIR] [MODEL NAME] [PRETRAINED_PATH] [[OTHER OPTIONS...]]
```


All samples in given directory.

```bash
$ python source_separation/synthesize.py test-dir [INPUT_DIR] [OUTPUT_DIR] [MODEL NAME] [PRETRAINED_PATH] [[OTHER OPTIONS...]]
```


## Experiments

- Reproduce experiments
  - General Voice Separation
    - single train code
    - Pretrained checkpoint is trained on default options / max_step 100000

  - Singing Voice Separation
    - joint train code
    - Pretrained checkpoint is trained on 4 GPUs, double (256) batch size.

  - Above options will be changed with curriculum learning and the other tries.

- Parameters and settings :
  It is tuned to find out good validation WSDR loss
  - refine_unet_base : 75M
  - refine_unet_larger : 95M

- Evaluation
  - To be filled with issue-9


## Loss curves

- Report again after curriculum learning.


## License

This repository is developed by [ILJI CHOI](https://github.com/Appleholic).  It is distributed under Apache License 2.0.

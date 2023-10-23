# Selective Transfer Learning of Cross-Modality Distillation for Monocular 3D Object Detection
## Introduction

In this paper, we systematically investigate the negative transfer problem induced by modality gap in cross-modality distillation for the first time, including the issues of feature overfitting and architecture inconsistency. 

## Usage

### Installation

This repo is tested on our local environment (python=3.8, cuda=11.3, pytorch=1.11), and we recommend you use anaconda to create a virtual environment:

```bash
conda create -n MonoSTL python=3.8
```

Then, activate the environment:

```bash
conda activate MonoSTL
```

Install  Install PyTorch:

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

and other  requirements: 

```bash
pip install -r requirements.txt
```


# Getting Started

## Training and Inference

* Download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

* Download the precomputed [depth maps]((https://drive.google.com/file/d/1qFZux7KC_gJ0UHEg-qGJKqteE9Ivojin/view?usp=sharing)) for the KITTI training set, which are provided by [CaDDN](https://github.com/TRAILab/CaDDN).

```
#ROOT
  |data/
    |KITTI/
      |ImageSets/ [already provided in this repo]
      |object/			
        |training/
          |calib/
          |image_2/
          |depth_2/
          |label_2/
        |testing/
          |calib/
          |image_2/
```

## train

```bash

nohup python /home/XXXX/code/MonoSTL/MonoSTL_MonoDLE/tools/train_val.py > test.txt 2>&1 &
```

# Acknowlegment

This repo benefits from the excellent work [MonoDLE](https://github.com/xinzhuma/monodle),[Monodistill](https://github.com/monster-ghost/MonoDistill),[MonoCon](https://github.com/2gunsu/monocon-pytorch),[MonoDETR](https://github.com/ZrrSkywalker/MonoDETR), [reviewKD](https://github.com/dvlab-research/ReviewKD).






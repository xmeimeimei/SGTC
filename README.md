# SGTC
## Introduction
This is the implementation of '[SGTC: Semantic-Guided Triplet Co-training for Sparsely Annotated Semi-Supervised Medical Image Segmentation](https://arxiv.org/pdf/2412.15526)'.
![](https://github.com/xmeimeimei/SGTC/blob/main/images/pipeline-AAAI.jpg)
## Requirements
Python == 3.8 and install from the ```requirements.txt``` using:
```
pip install -r requirements.txt
```
## Usage
Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data).
## Train the model
```
python train_SGTC.py --gpu 0 --dataset 'la' --split 'train'
```
## Acknowledgement
Part of the code is based on [UAMT](https://github.com/yulequan/UA-MT) and [CLIP-Driven Universal Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model) and [Desco](https://github.com/HengCai-NJU/DeSCO). Thanks for these authors for their valuable work.



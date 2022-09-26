# vizviva_fets_2022
Official PyTorch Code for [Hybrid Window Attention Based Transformer Architecture for Brain Tumor Segmentation](https://arxiv.org/pdf/2209.07704.pdf): Solution for FeTS 2022 Task 2.

## Data Preparation
Download data from [FeTS 2022 challenge page](https://www.synapse.org/#!Synapse:syn28546456/wiki/617246) and put it under data folder.

## Environment
Please prepare an environment with python=3.8, and then run the command "pip install -r requirements.txt" for the dependencies.

## Pre-Trained Weights
- Swin-T: https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth
- Download Swin-T pre-trained weights and add it under pretrained_ckpt folder

## Pre-Trained Model For FeTS 2022
- CR-Swin2-VT: https://drive.google.com/file/d/18ukXmZ5TzNgSUKco8PzPzPHUrkc510Z1/view?usp=sharing
- Download CR-Swin2-VT pre-trained model and add it under saved_model folder before running test.py

## Train/Test
- Train : Run the train script on BraTS 2021/FeTS 2022 Training Dataset  with Base model Configurations. 
```bash
python train.py 
```

- Test : Run the test script on FeTS 2022 Validation Dataset. 
```bash
python test.py 
```

## Acknowledgements
This repository makes liberal use of code from [open_brats2020](https://github.com/lescientifik/open_brats2020).

## Citing our work
```bash
@article{peiris2022hybrid,
  title={Hybrid Window Attention Based Transformer Architecture for Brain Tumor Segmentation},
  author={Peiris, Himashi and Hayat, Munawar and Chen, Zhaolin and Egan, Gary and Harandi, Mehrtash},
  journal={arXiv preprint arXiv:2209.07704},
  year={2022}
}
```




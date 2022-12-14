# Swin-Unet

## Introduction
This is a PyTorch(1.7.1) implementation of SwinViT-UNet for segmentation on the CrackSeg9k dataset. . The codes for the work "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation"(https://arxiv.org/abs/2105.05537). A validation for U-shaped Swin Transformer.

## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 1. Download pre-trained SwinViT-Unet model on the CrackSeg9K dataset
* [Get pre-trained model in this link] (https://drive.google.com/file/d/1zd8jcIWUuEyzQLWvUAOfl-drnlVNoUrK/view?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 2. Prepare data

- The datasets we used are provided by TransUnet's authors. Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License).

## 3. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test

- Train : To train on the CrackSeg9k dataset run

```bash
python3 experiment.py ##set max possible batch size from config 
```

- Test : To evaluate on the CrackSeg9K dataset run 

```bash
python3 test.py ##set max possible batch size from config 
```

## 5. Inference 

- For quick inference use the [Jupyter Notebook](https://github.com/Dhananjay42/crackseg9k/blob/main/SwinViT/inference.ipynb)



## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)



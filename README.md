# ComputerVision

This repository is made for the master degree's graduation thesis.
It composed with how to convert images and how to train models.
The details are below.

## ConvertImage
It contains how to convert binary files to jpg files.
There are two ways.
First one is applying "stream order method".
The other is applying "incremental coordinates method".

## PyTorchCNN
It conatins four models to train malware detection.
Theses are divided CNN, Transformer.
CNN models are EfficientNet and ConvNeXt(2022.01 SOTA).
Transformer models are ViT and SwinT.
All models are trained on Google Colab with PyTorch (Ubuntu 18.04).
Also all models check GradCAM with trained models.

## keras
It contains ViT Keras model to train malware detection.
This model trained on m1-tensorflow gpu.
Also model checks GradCAM with trained models.

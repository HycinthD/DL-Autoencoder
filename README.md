# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## THEORY
A convolutional autoencoder is a neural network that learns to compress (encode) an image into a lower‐dimensional representation and then reconstruct (decode) it back to its original size. By training on pairs of noisy and clean images, the encoder learns to extract robust feature maps that ignore noise, while the decoder uses these features to reconstruct the denoised image. Convolutional layers preserve spatial structure, making this approach well suited for image‐level tasks like denoising.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Problem Understanding and Dataset Selection

### STEP 2: 
 Preprocessing the Dataset
 
### STEP 3: 
Design the Convolutional Autoencoder Architecture

### STEP 4: 
Compile and Train the Model

### STEP 5: 
Evaluate the Model

### STEP 6: 
Visualization and Analysis

## PROGRAM

### Name: HYCINTH D

### Register Number: 212223240055

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor()
])

```
## RESULT
Thus, develop a convolutional autoencoder for image denoising application excuted succesfully

import os
from glob import glob
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np


from architecture import *
import unittest

"""
latent_layer_size = (1, 10, 4, 4)
num_attributes = 40

# Instantiate a sample tensor for the last layer in latent_layer
latent_layer= torch.randn(latent_layer_size)
print(latent_layer,"\n*******************\n")

#y = np.random.rand(1,40)
#y = np.array([[1,11],[2,22],[3,33],[4,44]])

y = np.array([1,2,3,4])

y = torch.tensor(y)
print(y.size())
print(y,"hey \n*******************\n")


y = y.view(1,-1)
print(y.size())
print(y,"\n*******************\n")


y = y.unsqueeze(2)
print(y.size())
print(y,"\n*******************\n")


y = y.unsqueeze(3)
print(y.size())
print(y,"\n*******************\n")

y=y.expand(-1, -1, 2, 2)
print(y.size())
print(y,"\n*******************\n")


# Concatenate along the channel dimension
decoder_input = torch.cat((latent_layer, y), dim=1)

# Print the size of the concatenated tensor
print(decoder_input.size())

# Tensor de taille (4, 2)
tensor = torch.randn(4, 2)

# Redimensionner le tensor
tensor_resized = tensor.view(4,2)
tensor_resized = tensor_resized.unsqueeze(0)
tensor_resized = tensor_resized.unsqueeze(2)

tensor_resized = tensor_resized.expand(-1, -1, 2, 2)

tensor_resized = tensor_resized .view(1, 4, 2, 2)

# Expand the tensor to (1, 4, 4, 4)
tensor_resized = tensor_resized.expand(-1, -1, 4, 4)
# Afficher la taille du tensor redimensionné
print(tensor_resized.size())
print(tensor_resized)

"""


def image_to_tensor( image_path =""):
        # Charger votre image (par exemple, avec PIL)
        image_path = r"C:\Users\Lynda\OneDrive\Documents\2023-2024\GitHub\Fader-Networks--Manipulating-Images-by-Sliding-Attributes\datasets\img_align_celeba\img_align_celeba\000001.jpg"
        image = Image.open(image_path)

        # Transformer l'image en un tensor
        transform = transforms.ToTensor()
        tensor_image = transform(image)

        # Ajouter une dimension pour représenter le lot (batch) si nécessaire
        tensor_image_batched = tensor_image.unsqueeze(0)

        # Afficher la taille du tensor résultant
        print(tensor_image_batched.size())

        plt.imshow(image)
        plt.axis('off')  # Turn off axis labels
        plt.show()


  


model = Encoder()
x_size = (1, 3, 256, 256)
# Instantiate a sample tensor 
x = torch.randn(x_size)

print(x.shape)
conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

res = conv1(x)

enc_output = model.encodage(x)

print(enc_output.shape)
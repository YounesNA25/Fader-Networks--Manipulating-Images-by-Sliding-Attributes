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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True))


        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))

            # Layer C64
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))

            # Layer C128
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))

            # Layer C256
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))

            # Layer C512
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))

            # Layer C512
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )


        self.encode_layers = [self.conv1,self.conv2,self.conv3,self.conv4,self.conv5,self.conv6,self.conv7]

    def encodage(self, x):
        outputs = [x]
        for layer in self.encode_layers:
            output = layer(outputs[-1])
            outputs.append(output)
        return outputs[-1]
    
    def forward(self,x):
        enc = self.encodage(x)
        return enc



class Decoder(nn.Module):
    def __init__(self, n_attributes):
        super(Decoder, self).__init__()
        self.n_attributes = n_attributes


        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512 + n_attributes, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

            # Layer TC512
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512 +  n_attributes, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

            # Layer TC256
        self.deconv3 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=256 +  n_attributes, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

            # Layer TC128
        self.deconv4 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=128 + n_attributes, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

            # Layer TC64
        self.deconv5 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=64 + n_attributes, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

            # Layer TC32
        self.deconv6 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=32 + n_attributes, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

            # Layer TC16
        self.deconv7 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=16 + n_attributes, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh())  # Typically used for normalizing image data between -1 and 1
        
        self.decode_layers = [self.deconv1,self.deconv2,self.deconv3,self.deconv4,self.deconv5,self.deconv6,self.deconv7]

    def decodage(self, latent_layers, y):
        y = y.view(-1,1)
        y = y.unsqueeze(2).unsqueeze(3)    
        batch_size = latent_layers.size(0)
        dec_outputs = latent_layers
        for layer in self.decode_layers:
            output_size = dec_outputs.size(2)
            input = torch.cat([dec_outputs, y.expand(batch_size, self.n_attributes, output_size, output_size)], dim = 1)
            dec_outputs = layer(input)
        return dec_outputs
    
    def forward(self,enc,y):
        dec = self.decodage(enc,y)
        return dec


class Descriminator(nn.Module):
    
    def __init__(self,n_attributes):
        self.n_attributes = n_attributes

        super(Descriminator,self).__init__()
        self.conv = nn.Sequential( 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )

        self.linear = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,self.n_attributes)
        )

    def forward(self, latent_representation):
        x = self.conv(latent_representation)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x
    

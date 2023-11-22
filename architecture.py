import os
from glob import glob
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import cv2

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
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True))


        conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))

            # Layer C64
        conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))

            # Layer C128
        conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))

            # Layer C256
        conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))

            # Layer C512
        conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))

            # Layer C512
        conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.encode_layers = [conv1,conv2,conv3,conv4,conv5,conv6,conv7]

    def encodage(self, x):
        outputs = [x]
        for layer in self.encode_layers:
            output = layer(outputs[-1])
            outputs.append(output)
        return outputs
    
    def forward(self,x):
        enc = self.encodage(x)
        return enc



class Decoder(nn.Module):
    def __init__(self, n_attributes):
        super(Decoder, self).__init__()
        self.n_attributes = n_attributes


        deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512 + n_attributes, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

            # Layer TC512
        deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512 +  n_attributes, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

            # Layer TC256
        deconv3 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=256 +  n_attributes, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

            # Layer TC128
        deconv4 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=128 + n_attributes, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

            # Layer TC64
        deconv5 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=64 + n_attributes, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

            # Layer TC32
        deconv6 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=32 + n_attributes, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

            # Layer TC16
        deconv7 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=16 + n_attributes, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh())  # Typically used for normalizing image data between -1 and 1
        
        self.decode_layers = [deconv1,deconv2,deconv3,deconv4,deconv5,deconv6,deconv7]

    def decodage(self, latent_layers, y):
        y = y.view(1,-1)
        y = y.unsqueeze(2).unsqueeze(3)    
        batch_size = latent_layers[0].size(0)
        dec_outputs = [latent_layers[-1]]
        for layer in self.decode_layers:
            output_size = dec_outputs[-1].size(2)
            input = torch.cat([dec_outputs[-1], y.expand(batch_size, self.n_attributes, output_size, output_size)], dim = 1)
            dec_outputs.append(layer(input))
        return dec_outputs
    
    def forward(self,enc,y):
        dec = self.decodage(enc,y)
        return dec



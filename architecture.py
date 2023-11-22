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
        """            
            Arguments:
                x : Input tensor of size (1, 512, 3, 3)   
            Returns:
                enc_output: Output tensor of size (1, 512, 2, 2)
        """
        enc_output = x
        for layer in self.encode_layers:
            enc_output = layer(enc_output)
        return enc_output
    
    def forward(self,x):
        enc = self.encodage(x)
        return enc



class Decoder(nn.Module):
    def __init__(self, n_attributes):
        super(Decoder, self).__init__()
        self.n_attributes = n_attributes


        deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512 + 2*n_attributes, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

            # Layer TC512
        deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512 +  2*n_attributes, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

            # Layer TC256
        deconv3 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=256 +  2*n_attributes, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

            # Layer TC128
        deconv4 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=128 + 2*n_attributes, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

            # Layer TC64
        deconv5 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=64 + 2*n_attributes, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

            # Layer TC32
        deconv6 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=32 + 2*n_attributes, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

            # Layer TC16
        deconv7 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=16 + 2*n_attributes, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh())  # Typically used for normalizing image data between -1 and 1
        
        self.decode_layers = [deconv1,deconv2,deconv3,deconv4,deconv5,deconv6,deconv7]
    

    """
    def decodage(self, latent_layer, y):  
        y = torch.tensor(y)
        y = y.view(1,-1)
        y = y.unsqueeze(2).unsqueeze(3)    
        batch_size = latent_layer.size(0)  # 1
        dec_outputs = [latent_layer]
        
        for layer in self.decode_layers:
            output_size = dec_outputs[-1].size(2)
            input = torch.cat([dec_outputs[-1], y.expand(batch_size, self.n_attributes, output_size, output_size)], dim = 1)
            dec_outputs.append(layer(input))
        return dec_outputs
    """

    def decodage(self, latent_layer, y):
            """            
            Arguments:
                latent_layer : torch.Tensor - Last layer of the decoder architecture = a tensor of size (1, 512, 2, 2)
                           y : np.ndarray   - A raw vector of size (1, nb_attributes) or (2, nb_attributes) ?
                
            Returns:
                torch.Tensor: Output tensor of size (1, 3, 256, 256)
            """
            
            y = torch.tensor(y)
            y = y.view(1, -1)   
            y = y.unsqueeze(2).unsqueeze(3) 

            batch_size = latent_layer.size(0)  #  1
            
            dec_output = [latent_layer]
            
            # Iterate through each layer in the decoder
            for layer in self.decode_layers:              
                # Concatenate the last output with y along the channel dimension
                input = torch.cat([dec_output, y.expand(batch_size, self.n_attributes, dec_output.size(2), dec_output.size(2))], dim=1)
                dec_output = layer(input)

            return dec_output



    def forward(self,enc,y):
        dec = self.decodage(enc,y)
        return dec



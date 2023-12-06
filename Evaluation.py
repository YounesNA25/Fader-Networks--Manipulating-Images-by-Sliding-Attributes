from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import os
#from logging import getLogger
#from torch.nn.utils import clip_grad_norm



    

class Evaluation:
    """
    A class responsible for evaluating an encoder, decoder, and discriminator model.
    Attributes:
        encoder: The encoder model to be evaluated.
        decoder: The decoder model to be evaluated.
        discriminator: The discriminator model to be evaluated.
        data_loader_val (DataLoader): DataLoader for validation data.
        data_loader: DataLoader for the evaluation dataset.
        use_cuda: Flag indicating whether to use GPU (CUDA).

    Methods:
        evaluation() : takes the number of epochs
    """

    def __init__(self, encoder, decoder, discriminator, data_loader_val, batch_size, use_cuda=True):

        self.encoder       = encoder
        self.decoder       = decoder
        self.discriminator = discriminator

        self.data_loader_val  = data_loader_val

        self.batch_size   = batch_size
        self.lambda_step  = 0.0001
        self.lambda_final = 500000
        self.n_tot_iter   = 0

        self.use_cuda = use_cuda

        # loss history
        self.history = {'AutoEncoder_loss':[], 'Adversial_loss':[], 'Discriminator_loss':[] }

   
        #self.step_msg = 'Train epoch {}\titeration {}\tlambda_e ={:.2e}\tadv. loss ={:.4f}\trec. loss ={:.4f}\tdsc. loss ={:.4f}\t'

    
    def evaluation(self):
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()

        for batch_x, batch_y in self.data_loader_val:
            if self.use_cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            Ex            = self.encoder(batch_x)
            reconstruct_x = self.decoder(Ex,batch_y[0][0])
            y_pred        = self.discriminator(Ex)

            # Flip y_ped : map (1) --> (0), and (0) --> (1)
            y_flipped = 1-batch_y[0][0]
            y_flipped = y_flipped.view(-1,1).float()

            # Compute losses
            reconstruct_loss = mse_loss(reconstruct_x, batch_x)
            
            discrim_loss = bce_loss(y_pred,y_flipped)

            lambdaE = self.lambda_step * float(min(self.n_tot_iter,self.lambda_final))/self.lambda_final
            advers_loss = reconstruct_loss  + lambdaE * discrim_loss

            self.n_tot_iter += 1

            self.history['Discriminator_loss'].append(discrim_loss.item())
            self.history['AutoEncoder_loss'].append(reconstruct_loss)
            self.history['Adversial_loss'].append(advers_loss)


    def display_progress(self):
        """
        # Display progress
        """

        autoencoder_losses = self.history['AutoEncoder_loss']
        adversarial_losses = self.history['Adversial_loss']
        discriminator_losses = self.history['Discriminator_loss']

        # Convert the lists to PyTorch tensors
        autoencoder_losses_tensor = torch.tensor(autoencoder_losses)
        adversarial_losses_tensor = torch.tensor(adversarial_losses)
        discriminator_losses_tensor = torch.tensor(discriminator_losses)


        # Compute the mean of each loss using PyTorch and convert to NumPy arrays
        mean_autoencoder_loss = autoencoder_losses_tensor.mean().item()
        mean_adversarial_loss = adversarial_losses_tensor.mean().item()
        mean_discriminator_loss = discriminator_losses_tensor.mean().item()

        del self.history['AutoEncoder_loss'][:]
        del self.history['Adversial_loss'][:]
        del self.history['Discriminator_loss'][:]

        return mean_autoencoder_loss, mean_adversarial_loss, mean_discriminator_loss
    
                        

                    
            
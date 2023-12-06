from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import json
import os
#from logging import getLogger
#from torch.nn.utils import clip_grad_norm



    

class Train:
    """
    A class responsible for training an encoder, decoder, and discriminator model.
    Attributes:
        encoder (nn.Module): The encoder model.
        decoder (nn.Module): The decoder model.
        discriminator (nn.Module): The discriminator model.
        data_loader_train (DataLoader): DataLoader for training data.
        batch_size (int): The size of each data batch.
        use_cuda: Flag indicating whether to use GPU (CUDA).
        encoder_optimizer (torch.optim.Adam): Optimizer for the encoder.
        decoder_optimizer (torch.optim.Adam): Optimizer for the decoder.
        discriminator_optimizer (torch.optim.Adam): Optimizer for the discriminator.

    Methods:
        discriminator_train(): Trains the discriminator model.
        autoencoder_train(): Trains the autoencoder (encoder and decoder) models.
    """

    def __init__(self, encoder, decoder, discriminator, data_loader_train, batch_size, use_cuda=True):

        
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        
        self.data_loader_train = data_loader_train

        self.batch_size = batch_size
        self.lambda_step = 0.0001
        self.lambda_final = 500000
        self.n_tot_iter = 0
        lr = 0.002
        betas = (0.5, 0.999)

        self.use_cuda = use_cuda

        self.encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, betas=betas)
        self.decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, betas=betas)
        self.discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)


        # loss history
        self.history = {'AutoEncoder_loss':[], 'Adversial_loss':[], 'Discriminator_loss':[] }


    def discriminator_train(self):
        # Set encoder and decoder to evaluation mode
        #     discriminator to training mode

        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.train()

        # Define the loss for the discriminator
        criterion = nn.BCEWithLogitsLoss()
        
        for batch_x, batch_y in self.data_loader_train:
            if self.use_cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()
           
            # Disable gradient calculation to save memory and avoid backpropagation
            with torch.no_grad():  
                enc = self.encoder(batch_x)

            y_pred = self.discriminator(enc)

            # Flip y_pred :  map (1) --> (0), and (0) --> (1)
            y_fake = 1-batch_y[0][0]
            y_fake = y_fake.view(-1,1).float()

            # Compute the discriminator loss
            #loss = criterion(y_pred.view(1,32).float(),batch_y[0][0].unsqueeze(0).float())
            loss = criterion(y_pred,y_fake)

            self.history['Discriminator_loss'].append(loss.item())
            
            # Zero gradients, perform backward pass, and update discriminator parameters
            self.discriminator_optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5) 

            # Update the discriminator parameters using the optimizer
            self.discriminator_optimizer.step()
            



    def autoencoder_train(self):
        # Set encoder and decoder to training mode
        #     discriminator to evaluation mode
        self.encoder.train()
        self.decoder.train()
        self.discriminator.eval()

        # Define loss functions
        mse_loss = nn.MSELoss()
        criterion = nn.BCEWithLogitsLoss()
        
        for batch_x, batch_y in self.data_loader_train:
            if self.use_cuda:
                batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            # Encode input and decode output
            enc_output = self.encoder(batch_x)
            dec_output = self.decoder(enc_output,batch_y[0][0])

            # Forward pass through discriminator
            y_pred = self.discriminator(enc_output)

            # Flip y_pred :  map (1) --> (0), and (0) --> (1)
            y_fake = 1-batch_y[0][0]
            y_fake = y_fake.view(-1,1).float()

            # Compute autoencoder loss from reconstruction
            reconstruct_loss = mse_loss(dec_output,batch_x)

            # Compute adversial loss
            lambdaE = self.lambda_step*float(min(self.n_tot_iter,self.lambda_final))/self.lambda_final
            advers_loss = reconstruct_loss + lambdaE*criterion(y_pred,y_fake)

            self.history['AutoEncoder_loss'].append(reconstruct_loss)
            self.history['Adversial_loss'].append(advers_loss)

            # Zero gradients, perform backward pass, and update encoder and decoder parameters
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            advers_loss.backward()

            # Clip gradients to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)
            
            # Update encoder and decoder parameters using their respective optimizers
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            self.n_tot_iter += 1
            


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

        
    """
    def save_model(self, history, history_directory, history_name, encoder_fpath, decoder_fpath, discriminator_fpath):
                     

        # Specify the history_directory and file name
        file_path = os.path.join(history_directory, history_name)

        # Create the history_directory if it doesn't exist
        if not os.path.exists(history_directory):
            os.makedirs(history_directory)

        # Save the history dictionary to the file
        with open(file_path, 'w') as file:
            json.dump(history, file)

        print(f"History data saved to {file_path}")
        
        print('Saving encoder parameters to %s' % (encoder_fpath))
        torch.save(self.encoder.state_dict(), encoder_fpath)
        print('Saving decoder parameters to %s' % (decoder_fpath))
        torch.save(self.decoder.state_dict(), encoder_fpath)
        print('Saving discriminator parameters to %s' % (discriminator_fpath))
        torch.save(self.discriminator.state_dict(), discriminator_fpath)
    """
    def save_model(self, history, history_directory, history_name, encoder_fpath, decoder_fpath, discriminator_fpath):
        """
        Save the model and loss history
        """                       

        # Specify the history_directory and file name
        file_path = os.path.join(history_directory, history_name)

        # Create the history_directory if it doesn't exist
        if not os.path.exists(history_directory):
            os.makedirs(history_directory)

        # Save the history dictionary to the file
        with open(file_path, 'w') as file:
            json.dump(history, file)

        print(f"History data saved to {file_path}")

        # Save model parameters
        self.save_model_parameters(self.encoder, encoder_fpath)
        self.save_model_parameters(self.decoder, decoder_fpath)
        self.save_model_parameters(self.discriminator, discriminator_fpath)

    def save_model_parameters(self, model, filepath):
        print(f'Saving {model.__class__.__name__} parameters to {filepath}')
        torch.save(model.state_dict(), filepath)


            
    







        


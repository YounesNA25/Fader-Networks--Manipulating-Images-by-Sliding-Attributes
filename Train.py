from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
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
        data_loader_test (DataLoader): DataLoader for test data.
        data_loader_val (DataLoader): DataLoader for validation data.
        batch_size (int): The size of each data batch.
        encoder_optimizer (torch.optim.Adam): Optimizer for the encoder.
        decoder_optimizer (torch.optim.Adam): Optimizer for the decoder.
        discriminator_optimizer (torch.optim.Adam): Optimizer for the discriminator.

    Methods:
        discriminator_train(): Trains the discriminator model.
        autoencoder_train(): Trains the autoencoder (encoder and decoder) models.
    """

    def __init__(self, encoder, decoder, discriminator, data_loader_train, batch_size):

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

        self.encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, betas=betas)
        self.decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, betas=betas)
        self.discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    
        self.step_msg = 'Train epoch {}\titeration {}\tlambda_e ={:.2e}\tadv. loss ={:.4f}\trec. loss ={:.4f}\tdsc. loss ={:.4f}\t'

    def discriminator_train(self):
        self.encoder.eval()
        self.decoder.eval()
        criterion = nn.BCEWithLogitsLoss()
        self.discriminator.train()
        for batch_x, batch_y in self.data_loader_train:
            #batch_x,batch_y = batch_x.cuda(),batch_y.cuda()
            print('111')
            with torch.no_grad():  #bloc no_grad pour économiser de la mémoire en ne maintenant pas les calculs de la rétropropagation
                enc = self.encoder(batch_x)
            predicted_y = self.discriminator(enc)
            loss = criterion(predicted_y.view(1,32).float(),batch_y[0][0].unsqueeze(0).float())
            #self.stats['discriminator_costs'].append(loss.item())
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5)  #pour empêcher l'explosion du gradient
            self.discriminator_optimizer.step()


    def autoencoder_train(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.eval()
        mse_loss = nn.MSELoss()
        criterion = nn.BCEWithLogitsLoss()
        self.discriminator.train()
        for batch_x, batch_y in self.data_loader_train:
            #batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            # encode / decode
            enc_output = self.encoder(batch_x)
            dec_output = self.decoder(enc_output,batch_y[0][0])

            # autoencoder loss from reconstruction
            loss = mse_loss(dec_output,batch_x)
            lambdaE = self.lambda_step*float(min(self.n_tot_iter,self.lambda_final))/self.lambda_final
 
            y_pred = self.discriminator(enc_output)
            y_fake = 1-batch_y[0][0]
            y_fake = y_fake.view(-1,1).float()

            loss = loss + lambdaE*criterion(y_pred,y_fake)
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            self.n_tot_iter += 1
            
    
    def evaluation(self, epoch):
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()

        for batch_x, batch_y in self.data_loader_val:
            #if self.use_cuda:
                #batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

            Ex            = self.encoder(batch_x)
            reconstruct_x = self.decoder(Ex,batch_y[0][0])
            y_pred        = self.discriminator(Ex)

            # Calculate losses
            reconstruct_loss = mse_loss(reconstruct_x, batch_x)
            
            # map (1) --> (0), and (0) --> (1)
            y_flipped = 1-batch_y[0][0]
            y_flipped = y_flipped.view(-1,1).float()

            discrim_loss = bce_loss(y_pred,y_flipped)
            lambdaE = self.lambda_step * float(min(self.n_tot_iter,self.lambda_final))/self.lambda_final
            advers_loss = reconstruct_loss  + lambdaE * discrim_loss

            self.n_tot_iter += 1

            ### Display progress
            print(self.step_msg.format(epoch,self.n_tot_iter,lambdaE,advers_loss.item(), reconstruct_loss.item(), discrim_loss.item()),
                    end='\r',flush=True)
    
                        

                    
            # Show result
            flipped_x     = self.decoder(Ex,y_flipped)
            src_image = (batch_x.data.numpy().transpose(0, 2, 3, 1) + 1) / 2
            rec_image = (reconstruct_x.data.numpy().transpose(0, 2, 3, 1) + 1) / 2
            flp_image = (flipped_x.data.numpy().transpose(0, 2, 3, 1) + 1) / 2

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(src_image[0])
            axes[0].set_title('Original')

            axes[1].imshow(rec_image[0])
            axes[1].set_title('Reconstructed (Correct Y)')

            axes[2].imshow(flp_image[0])
            axes[2].set_title('Reconstructed (Flipped Y)')

            plt.show()
            break

            ### Test 
            if True:
                self.encoder.eval()
                self.decoder.eval()

                for batch_x_test, batch_y_test in enumerate(self.data_loader_test):
                    #if self.use_cuda:
                        #batch_x,batch_y = batch_x.cuda(),batch_y.cuda()

                    # randomly choose an attribute and swap the targets
                    #to_swap = np.random.choice(self.data_loader_test.attr_labels )
                    #swap_idx, = np.where(self.data_loader_test.attr_labels == to_swap)[0]

                    # map (1) --> (0), and (0) --> (1)
                    y_flipped = 1-batch_y_test[0][0]
                    y_flipped = y_flipped.view(-1,1).float()

                    Ex            = self.encoder(batch_x_test)
                    reconstruct_x = self.decoder(Ex,batch_y_test[0][0])
                    flipped_x     = self.decoder(Ex,y_flipped)

                    ### Display progress
                    print(self.step_msg.format(epoch,self.n_tot_iter,lambdaE,advers_loss.item(), reconstruct_loss.item(), discrim_loss.item()),
                    end='\r',flush=True)
                            
                    # Show result
                    src_image = (batch_x_test.data.numpy().transpose(0, 2, 3, 1) + 1) / 2
                    rec_image = (reconstruct_x.data.numpy().transpose(0, 2, 3, 1) + 1) / 2
                    flp_image = (flipped_x.data.numpy().transpose(0, 2, 3, 1) + 1) / 2

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(src_image[0])
                    axes[0].set_title('Original')

                    axes[1].imshow(rec_image[0])
                    axes[1].set_title('Reconstructed (Correct Y)')

                    axes[2].imshow(flp_image[0])
                    axes[2].set_title('Reconstructed (Flipped Y)')

                    plt.show()
                    break


        


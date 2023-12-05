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
            
  
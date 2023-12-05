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

    def __init__(self, encoder, decoder, discriminator, data_loader_test, data_loader_val, batch_size):

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.data_loader_test = data_loader_test
        self.data_loader_val = data_loader_val
        self.batch_size = batch_size
        self.lambda_step = 0.0001
        self.lambda_final = 500000
        self.n_tot_iter = 0
   
        self.step_msg = 'Train epoch {}\titeration {}\tlambda_e ={:.2e}\tadv. loss ={:.4f}\trec. loss ={:.4f}\tdsc. loss ={:.4f}\t'

            
    
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


        


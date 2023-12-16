import os
import numpy as np
import torch
import torch.nn as nn
from data.dataset import Datasets
from src.architecture import Encoder, Decoder, Discriminator
import argparse
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt




def parse_args():
    """
    Parse les arguments pour la configuration de l'entraînement.
    """
    parser = argparse.ArgumentParser(description="Train Fader_Network On dataset")
    parser.add_argument("--root-rszimages", help="Directory path to resized images.", default="resized_images/", type=str)
    parser.add_argument("--root-attributes", help="Directory path to processed attributes.", default="processed_attributes", type=str)
    parser.add_argument("--attr-chg", help="Attributes to change.", default=['Smiling'], type=list)
    parser.add_argument("--batch-size", default = 32, type=int, help="Batch size")    
    parser.add_argument("--epochs-max", default = 1000, type=int, help="Epochs of train loop")
    parser.add_argument("--save-interval", default=10, type=int, help="Interval of epochs to save model checkpoints")
    parser.add_argument("--checkpoint-dir", default="checkpoints_smile_sigmo/", type=str, help="Directory to save model checkpoints")
    parser.add_argument("--encoder-checkpoint", default="", type=str, help="Chemin vers le fichier de point de contrôle de l'encodeur pour reprendre l'entraînement")
    parser.add_argument("--decoder-checkpoint", default="", type=str, help="Chemin vers le fichier de point de contrôle du décodeur pour reprendre l'entraînement")
    parser.add_argument("--discriminator-checkpoint", default="", type=str, help="Chemin vers le fichier de point de contrôle du discriminateur pour reprendre l'entraînement")
    
    args = parser.parse_args()
    return args
        

def build_train_data_loader(args):
    """
    Train DataLoader.
    """
    train_dataset = Datasets(
        root_images=args.root_rszimages, 
        root_attributes=args.root_attributes,
        attributes=args.attr_chg,
        chunk = 'train',
    )
    return torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1,
        pin_memory=True
    )
        
def build_val_data_loader(args):
    """
    Validation DataLoader.
    """
    val_dataset = Datasets(
        root_images=args.root_rszimages, 
        root_attributes=args.root_attributes,
        attributes=args.attr_chg,
        chunk = 'val',
    )
    return torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1,
        pin_memory=True
    )

def initialize_networks(use_cuda):
    """
    Init les models (encodeur, décodeur, discriminateur).
    """
    encoder = Encoder()
    decoder = Decoder(1)
    discriminator = Discriminator(1)
    if use_cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
    return encoder, decoder, discriminator

def load_checkpoints(encoder, decoder, discriminator, args, use_cuda):
    """
    Loade les models
    """
    if args.encoder_checkpoint and os.path.isfile(args.encoder_checkpoint):
        encoder.load_state_dict(torch.load(args.encoder_checkpoint))
        if use_cuda:
            encoder.cuda()
        print(f"encodeur loaded from {args.encoder_checkpoint}")

    if args.decoder_checkpoint and os.path.isfile(args.decoder_checkpoint):
        decoder.load_state_dict(torch.load(args.decoder_checkpoint))
        if use_cuda:
            decoder.cuda()
        print(f"décodeur loaded from {args.decoder_checkpoint}")

    if args.discriminator_checkpoint and os.path.isfile(args.discriminator_checkpoint):
        discriminator.load_state_dict(torch.load(args.discriminator_checkpoint))
        if use_cuda:
            discriminator.cuda()
        print(f"discriminateur loaded from {args.discriminator_checkpoint}")
        
    return encoder, decoder, discriminator

def live_plot(data_dict, figsize=(7,5), title=''):
    """
    Fonction pour mettre à jour séquentiellement un graphique.
    """
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        plt.plot(data, label=label)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='center left') # the plot evolves to the right
    plt.show()
    
    
def train_one_epoch(encoder, decoder, discriminator, train_data_loader, n_tot_iter, lambda_step, lambda_final, encoder_optimizer, decoder_optimizer, discriminator_optimizer, mse_loss, bce_loss, use_cuda):
    """
    Entraîne le model pour une epoch
    """
    encoder.train()
    decoder.train()
    discriminator.train()

    total_reconstruct_loss = 0.0
    total_advers_loss = 0.0
    total_dis_loss = 0.0
    
    for batch_x, batch_y in tqdm(train_data_loader, desc="Training Batch"):
        if use_cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        
        enc_output = encoder(batch_x)
        dec_output = decoder(enc_output, batch_y[:, 0])
        
        y_pred = discriminator(enc_output)
        y_fake = 1 - batch_y[:, 0]
        y_fake = y_fake.view(-1, 1).float()
        
        reconstruct_loss = mse_loss(dec_output, batch_x)
        lambdaE = float(min(n_tot_iter, lambda_final))/lambda_final
        advers_loss = reconstruct_loss +  lambdaE*bce_loss(y_pred, y_fake)

        total_reconstruct_loss += reconstruct_loss.item()
        total_advers_loss += advers_loss.item()
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        advers_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        enc_output.detach_()
        y_pred = discriminator(enc_output)
        dis_loss = bce_loss(y_pred, batch_y[:, 0].view(-1, 1).float())
        
        total_dis_loss += dis_loss.item()
        
        discriminator_optimizer.zero_grad()
        dis_loss.backward()
        discriminator_optimizer.step()

        n_tot_iter += 1

    avg_reconstruct_loss = total_reconstruct_loss / len(train_data_loader)
    avg_advers_loss = total_advers_loss / len(train_data_loader)
    avg_dis_loss = total_dis_loss / len(train_data_loader)
    return avg_reconstruct_loss, avg_advers_loss, avg_dis_loss

def eval_one_epoch(encoder, decoder, discriminator,eval_data_loader, n_tot_iter, lambda_step, lambda_final, mse_loss, bce_loss, use_cuda):
    """
    Évalue le model pour une epoch.
    """
    encoder.eval()
    decoder.eval()
    discriminator.eval()
        
    total_reconstruct_loss = 0.0
    total_advers_loss = 0.0
    total_dis_loss = 0.0
    for batch_x, batch_y in tqdm(eval_data_loader, desc="Evaluation Batch"):
        if use_cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        
        enc_output = encoder(batch_x)
        dec_output = decoder(enc_output, batch_y[:, 0])
        
        y_pred = discriminator(enc_output)
        y_fake = 1 - batch_y[:, 0]
        y_fake = y_fake.view(-1, 1).float()
        
        reconstruct_loss = mse_loss(dec_output, batch_x)
        lambdaE = float(min(n_tot_iter, lambda_final))/lambda_final
        advers_loss = reconstruct_loss + lambdaE * bce_loss(y_pred, y_fake)

        total_reconstruct_loss += reconstruct_loss.item()
        total_advers_loss += advers_loss.item()
     

        # enc_output.detach_()
        y_pred = discriminator(enc_output)
        dis_loss = bce_loss(y_pred, batch_y[:, 0].view(-1, 1).float())
        
        total_dis_loss += dis_loss.item()

        n_tot_iter += 1

    avg_reconstruct_loss = total_reconstruct_loss / len(eval_data_loader)
    avg_advers_loss = total_advers_loss / len(eval_data_loader)
    avg_dis_loss = total_dis_loss / len(eval_data_loader)
    return avg_reconstruct_loss, avg_advers_loss, avg_dis_loss



def train_model(args, use_cuda=True):
    """
    Fonction principale pour entraîner le modèle.
    """
    encoder, decoder, discriminator = initialize_networks(use_cuda)
    
    train_data_loader = build_train_data_loader(args)
    eval_data_loader = build_val_data_loader(args)
    lambda_step = 0.0001
    lambda_final = 500000
    n_tot_iter = 0
    lr = 0.0002
    betas = (0.5, 0.999)
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, betas=betas)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, betas=betas)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    to_plot = {'dis loss': [], 'reconstruct loss': [], 'advers loss': []}
    
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    for n_epoch in range(args.epochs_max):
        
        avg_reconstruct_loss, avg_advers_loss, avg_dis_loss = train_one_epoch(encoder, decoder, discriminator, train_data_loader, n_tot_iter, lambda_step, lambda_final, encoder_optimizer, decoder_optimizer, discriminator_optimizer, mse_loss, bce_loss, use_cuda)
        
        avg_reconstruct_loss_eval, avg_advers_loss_eval, avg_dis_loss_eval = eval_one_epoch(encoder, decoder, discriminator, eval_data_loader, n_tot_iter, lambda_step, lambda_final, mse_loss, bce_loss, use_cuda)

        
        to_plot['dis loss'].append(avg_dis_loss)
        to_plot['reconstruct loss'].append(avg_reconstruct_loss)
        to_plot['advers loss'].append(avg_advers_loss)
        live_plot(to_plot)
        
        
        reconstruct_loss_file = 'reconstruct_loss_.txt'
        advers_loss_file = 'advers_loss_.txt'
        dis_loss_file = 'dis_loss_.txt'
        
        reconstruct_loss_eval_file = 'reconstruct_loss_eval_.txt'
        advers_loss_eval_file = 'advers_loss_eval_.txt'
        dis_loss_eval_file = 'dis_loss_eval_.txt'
        losses_file = 'losses_file.txt'
        
        with open(losses_file, 'a') as file:
            file.write(f'rec. loss: {avg_reconstruct_loss:.6f}\t rec. val: {avg_reconstruct_loss_eval:.6f}\t adv. loss: {avg_advers_loss:.6f}\t adv. val: {avg_advers_loss_eval:.6f}\t dsc. loss: {avg_dis_loss:.6f}\t dsc. val: {avg_dis_loss_eval:.6f}\n')
        
        print(f"Epoch [{n_epoch + 1}/{args.epochs_max}]: Train Rec. loss: {avg_reconstruct_loss:.4f}, Adv. loss: {avg_advers_loss:.4f}, Dis. Loss: {avg_dis_loss:.4f}")                 
        print(f"Validation Rec. loss: {avg_reconstruct_loss_eval:.4f}, Adv. loss: {avg_advers_loss_eval:.4f}, Dis. Loss: {avg_dis_loss_eval:.4f}")
        
        if (n_epoch + 1) % args.save_interval == 0:
            checkpoint_enc = os.path.join(checkpoint_dir, f'encoder_epoch_{n_epoch + 1}.pt')
            checkpoint_dec = os.path.join(checkpoint_dir, f'decoder_epoch_{n_epoch + 1}.pt')
            checkpoint_dis = os.path.join(checkpoint_dir, f'discriminator_epoch_{n_epoch + 1}.pt')
            
            torch.save(encoder.state_dict(), checkpoint_enc)
            torch.save(decoder.state_dict(), checkpoint_dec)
            torch.save(discriminator.state_dict(), checkpoint_dis)
            print(f"Epoch [{n_epoch + 1}] checkpoint saved")

if __name__ == "__main__":
    args = parse_args()
    train_model(args)

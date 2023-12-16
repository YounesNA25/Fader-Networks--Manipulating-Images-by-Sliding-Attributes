import os
import torch
from data import *
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.colors as mcolors

def split_train_val_test(attributes_names):

    train_set = Datasets(
        root_images='.\data\img_align_celeba_Preprocessed', 
        root_attributes='.\data\processed_attributes',
        attributes=attributes_names,
        chunk = 'train')
    
    test_set  = Datasets(root_images='.\data\img_align_celeba_Preprocessed',
                         root_attributes='.\data\processed_attributes',
                         attributes=attributes_names,
                         chunk = 'test')
    
    val_set   = Datasets(root_images='.\data\img_align_celeba_Preprocessed', 
                         root_attributes='.\data\processed_attributes',
                         attributes=attributes_names,
                         chunk = 'val')
    
    return train_set, test_set, val_set


def plot_training_loss(history):

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(history['ae_loss'], label="training loss",color='blue')
    axs[0].plot(history['ae_val_loss'], label="validation loss",color='red')
    axs[0].set_xlabel("Loss")
    axs[0].set_ylabel("Epochs")
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title("Auto_encoder : MSE loss")

    axs[1].plot(history['dsc_loss'], label="training loss",color='green')
    axs[1].plot(history['dsc_val_loss'], label="validation loss",color='orange')
    axs[1].set_xlabel("Loss")
    axs[1].set_ylabel("Epochs")
    axs[1].legend()
    axs[1].grid()
    axs[1].set_title("Discriminator : BCE loss")

    plt.tight_layout()
    plt.show()


    
    
def plot_accuracy(train_acc, valid_acc):

    num_epochs = len(train_acc)

    plt.plot(np.arange(1, num_epochs+1), 
             train_acc, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    
# -------------------------------------------------------------------------------------------

"""
def load_model_histroy(file, train = False, params=params):
    
    #Load a model
    #-----  
    #Parameters: 
    #file: str, filename
    
    path = str(Path(__file__).parent)+'/'+params.get("MODELS_PATH")
    model_params = np.load(path+'/'+file+'/params.npy', allow_pickle = True).item()
    history = np.load(path+'/'+file+'/history.npy', allow_pickle = True).item()

    model_type = file.split('_')[0]

    if model_type == 'classifier':
        model = Classifier(model_params, history)
    elif model_type =='fader':
        model = Fader(model_params)
    elif model_type =='ae':
        model = AutoEncoder(model_params)
    else: 
        raise ValueError(f"invalid model_type = {model_type}, possible value are 'classifier', 'fader' or 'ae'")

    model.load_weights(path+'/'+file+'/weights')
    model.trainable = train

    return model, history
"""

# -------------------------------------------------------------------------------------------

def show_images_attributes_celaba_2(data, attributes_names, num_images=10):

    if num_images == 1:

        # Show the first image
        img, attributes = data[0]
        # Reverse normalization
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        reverse_normalize = transforms.Normalize(-mean / std, 1.0 / std)
        img_reverse_normalized = reverse_normalize(img)
        img_transformed = transforms.ToPILImage()(img_reverse_normalized )
        label = " ".join(f"{name}={value[0]}" for name, value in zip(attributes_names, attributes))
        plt.imshow(img_transformed)
        plt.title(label,fontsize=5)
        plt.axis('off')
    
    else : 

        # Determine the number of rows and columns for the subplot grid
        num_rows = int(num_images ** 0.5)
        num_cols = (num_images // num_rows) + (num_images % num_rows)

        # Create a subplot grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        axes = axes.ravel()

        # Iterate through the data and display each image in a subplot
        for id in range(num_images):
            img, attributes_Onehot, attributes = data[id]
            # Reverse normalization
            mean = torch.tensor([0.5, 0.5, 0.5])
            std = torch.tensor([0.5, 0.5, 0.5])
            reverse_normalize = transforms.Normalize(-mean / std, 1.0 / std)
            img_reverse_normalized = reverse_normalize(img)
            img_transformed = transforms.ToPILImage()(img_reverse_normalized )
            label = " ".join(f"{name}={value}" for name, value in zip(attributes_names, attributes))
            axes[id].imshow(img_transformed)
            axes[id].set_title(label,fontsize=5)
            axes[id].axis('off')

        # Hide any empty subplots
        for i in range(num_images, num_rows * num_cols):
            fig.delaxes(axes[i])


        # Adjust layout for better spacing
        plt.tight_layout()

    plt.show()


def show_images_attributes_celaba(data, attributes_names, num_images=10):

    if num_images == 1:

        # Show the first image
        img, attributes = data[0]
        # Reverse normalization
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        reverse_normalize = transforms.Normalize(-mean / std, 1.0 / std)
        img_reverse_normalized = reverse_normalize(img)
        img_transformed = transforms.ToPILImage()(img_reverse_normalized )
        label = " ".join(f"{name}={value[0]}" for name, value in zip(attributes_names, attributes))
        plt.imshow(img_transformed)
        plt.title(label,fontsize=5)
        plt.axis('off')
    
    else : 

        # Determine the number of rows and columns for the subplot grid
        num_rows = int(num_images ** 0.5)
        num_cols = (num_images // num_rows) + (num_images % num_rows)

        # Create a subplot grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        axes = axes.ravel()

        # Iterate through the data and display each image in a subplot
        for id in range(num_images):
            img, attributes = data[id]
            # Reverse normalization
            mean = torch.tensor([0.5, 0.5, 0.5])
            std = torch.tensor([0.5, 0.5, 0.5])
            reverse_normalize = transforms.Normalize(-mean / std, 1.0 / std)
            img_reverse_normalized = reverse_normalize(img)
            img_transformed = transforms.ToPILImage()(img_reverse_normalized )
            label = " ".join(f"{name}={value[0]}" for name, value in zip(attributes_names, attributes))
            axes[id].imshow(img_transformed)
            axes[id].set_title(label,fontsize=5)
            axes[id].axis('off')

        # Hide any empty subplots
        for i in range(num_images, num_rows * num_cols):
            fig.delaxes(axes[i])


        # Adjust layout for better spacing
        plt.tight_layout()

    plt.show()

# -------------------------------------------------------------------------------------------

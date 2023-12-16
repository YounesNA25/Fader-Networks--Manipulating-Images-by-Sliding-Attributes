import argparse
import json
import os
import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.architecture import Decoder, Discriminator, Encoder
from data.dataset import Datasets


def parse_args():
    parser = argparse.ArgumentParser(description="CelebA dataset")   

    parser.add_argument(
        "--root-rezimages",
        help="Directory path to resized images.",
        default="./datafolder/resized_images/",
        type=str,
    )
    parser.add_argument(
        "--root-attributes",
        help="Directory path to processed attributes.",
        default="./datafolder/processed_attributes",
        type=str,
    )
    parser.add_argument(
        "--root-names",
        help="Directory path to the text file of attribute names.",
        default='./datafolder/attribute_names.txt',
        type=str,
    )
    parser.add_argument(
        "--num-images",
        help="Number of images to show.",
        default=45,
        type=int,
    )
    parser.add_argument(
        "--to-show-hist",
        help="Bool to show histogram of attributes",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--to-show-img",
        help="Bool to show grid of dataset images",
        default=False,
        type=bool,
    )

    args = parser.parse_args()
    return args

def build_dataset(args,attribute_names):
    return Datasets(
        root_images=args.root_rezimages,
        root_attributes=args.root_attributes,
        attributes=attribute_names,
        chunk="",
    )

def read_attribute_names(arg):
    attribute_names = []

    with open(arg.root_names, 'r') as file:
        for nb, line in enumerate (file)   :
    
                # Split the line into index and attribute name
                parts = line.strip().split(' - ')
                if len(parts) == 2:
                    attribute_names.append(parts[1])

    return attribute_names


def show_images_attributes_celaba(arg, data, attribute_names):

    if arg.num_images == 1:
        # Show the first image
        img, attributes = data[0]
        print(attributes)
        # Reverse normalization
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        reverse_normalize = transforms.Normalize(-mean / std, 1.0 / std)
        img_reverse_normalized = reverse_normalize(img)
        img_transformed = transforms.ToPILImage()(img_reverse_normalized)
        label = "\n".join(
            f"{name}={value[0]}" for name, value in zip(attribute_names, attributes)
        )
        plt.imshow(img_transformed)
        plt.title(label, fontsize=5)
        plt.axis("off")

    else:
        # Determine the number of rows and columns for the subplot grid
        num_images_per_row = 15

        # Determine the number of rows and columns for the subplot grid
        num_rows = (arg.num_images + num_images_per_row - 1) // num_images_per_row
        num_cols = min(arg.num_images, num_images_per_row)


        # Create a subplot grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 15))
        axes = axes.ravel()

        # Iterate through the data and display each image in a subplot
        for id in range(arg.num_images):
            img, attributes = data[id]
            print(attribute_names)
            
            # Reverse normalization
            mean = torch.tensor([0.5, 0.5, 0.5])
            std = torch.tensor([0.5, 0.5, 0.5])
            reverse_normalize = transforms.Normalize(-mean / std, 1.0 / std)
            img_reverse_normalized = reverse_normalize(img)
            img_transformed = transforms.ToPILImage()(img_reverse_normalized)
            label = "\n".join(
                name
                for name, value in zip(attribute_names, attributes)

                if value[0] == 1
            )
            axes[id].imshow(img_transformed)
            axes[id].set_title(label, fontsize=5)
            axes[id].axis("off")

        # Hide any empty subplots
        for i in range(arg.num_images, num_rows * num_cols):
            fig.delaxes(axes[i])

        # Adjust layout for better spacing
        plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def histogram_of_attributes(arg, data, attribute_names):
    nb_appearances_train = np.zeros(len(attribute_names))
    nb_appearances_val = np.zeros(len(attribute_names))
    nb_appearances_test = np.zeros(len(attribute_names))

    total_data = len(data)

    for nb_data, (img, attributes) in tqdm(enumerate(data, 1), total=total_data, desc="Histogram progress"):
        for id, value in enumerate(attributes):
            if value[0] == 1 and nb_data < 162770:
                nb_appearances_train[id] += 1
            elif value[0] == 1 and 162770 <= nb_data < 182637:
                nb_appearances_val[id] += 1
            elif value[0] == 1:
                nb_appearances_test[id] += 1

    # Choose different shades of blue, violet, and green
    color_train = 'lightblue'
    color_val = 'lightcoral'
    color_test = 'lightgreen'

    # Plot the histogram using the actual data
    plt.bar(attribute_names, nb_appearances_train, color=color_train, label='Train')
    plt.bar(attribute_names, nb_appearances_val, color=color_val, bottom=nb_appearances_train, label='Validation')
    plt.bar(attribute_names, nb_appearances_test, color=color_test, bottom=nb_appearances_train + nb_appearances_val, label='Test')

    # Display the sum on each section of the bar
    for i in range(len(attribute_names)):
        plt.text(i, nb_appearances_train[i] + nb_appearances_val[i] + nb_appearances_test[i], 
                 str(int(nb_appearances_train[i] + nb_appearances_val[i] + nb_appearances_test[i])), 
                 ha='center', va='bottom')

    plt.ylabel('Number of Appearances')
    plt.title('Histogram of Attributes')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.legend()
    plt.show()

                         
def celeba_dataset(args):

    attribute_names = read_attribute_names(args)

    data = build_dataset(args,attribute_names) 

    if args.to_show_img:
        show_images_attributes_celaba(args, data, attribute_names)
    if args.to_show_hist:
        histogram_of_attributes(args, data, attribute_names)


if __name__ == "__main__":
    args = parse_args()
    celeba_dataset(args)

    

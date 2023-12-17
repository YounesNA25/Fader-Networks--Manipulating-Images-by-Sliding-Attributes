import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot losses")
    parser.add_argument(
        "--fpath",
        help="path to losses.txt",
        default='trained models\eyeglasses_losses_file.txt', 
        type=str,
    )
    args = parser.parse_args()
    return args

def read_file(arg):
    # Read data from the .txt file
    with open(arg.fpath, 'r') as file:
        lines = file.readlines()

    # Parse the values from each line
    rec_loss = []
    rec_val_loss = []
    adv_loss = []
    adv_val_loss = []
    dsc_loss = []
    dsc_val_loss = []

    for line in lines:
        if line != '\n':
            values = line.split('\t')
            rec_loss.append(float(values[1].split(':')[1].strip()))
            rec_val_loss.append(float(values[1].split(':')[1].strip()))
            dsc_loss.append(float(values[4].split(':')[1].strip()))
            dsc_val_loss.append(float(values[5].split(':')[1].strip()))

    return rec_loss, rec_val_loss, dsc_loss, dsc_val_loss

def plot_losses(arg):
    # Read data from the .txt file
    rec_loss, rec_val_loss, dsc_loss, dsc_val_loss = read_file(arg)

    # Extract the title from the file path
    titles_1 = arg.fpath.split('\\')  
    titles_2 = titles_1[1].split('.')[0]
    title = titles_2.split('_')[0]


    # Create a subplot with two columns
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot rec and val rec on the left
    axs[0].plot(rec_loss, label='Rec. Loss', color='lightblue')
    axs[0].plot(rec_val_loss, label='Rec. Val Loss', color='lightcoral')
    axs[0].set_title('Rec. and Val Rec. Losses')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot disc and val disc on the right
    axs[1].plot(dsc_loss, label='Dsc. Loss', color='lightblue')
    axs[1].plot(dsc_val_loss, label='Dsc. Val Loss', color='lightcoral')
    axs[1].set_title('Dsc. and Val Dsc. Losses')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)


    # Adjust layout
    plt.tight_layout()

    # Save the figure
    save_dir = f".\Results"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    plt.savefig(
        f".\Results\{title}_losses.png"
    )

    # Show the plot
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    plot_losses(args)

import argparse
import json
import os
import random
import sys
import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from data.dataset import Datasets
from src.architecture import Decoder, Discriminator, Encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Test Fader_Network On dataset")
    parser.add_argument(
        "--use-cuda",
        help="Use CUDA.",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--root-rezimages",
        help="Directory path to resized images.",
        default="datafolder/resized_images",
        type=str,
    )
    parser.add_argument(
        "--root-attributes",
        help="Directory path to processed attributes.",
        default="datafolder/processed_attributes",
        type=str,
    )
    parser.add_argument(
        "--attr-chg",
        help="Attributes to change.",
        default=["Smiling"],
        type=list,
    )
    parser.add_argument(
        "--nb-alpha",
        help="Number of weights of an attribute.",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--nb-attributes2flip",
        help="Number of attributes to change.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--nb-x2flip",
        help="Number of images to change.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--sample-batch",
        default=20,
        type=int,
        help="The frequency for displaying and saving results.",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--encoder-fpath",
        default="trained models/encoder_smiling_60.pt",
        type=str,
        help="file path to load the encoder",
    )
    parser.add_argument(
        "--decoder-fpath",
        default="trained models/decoder_smiling_60.pt",
        type=str,
        help="file path to load the decoder",
    )
    parser.add_argument(
        "--discriminator-fpath",
        default="trained models/discriminator_smiling_60.pt",
        type=str,
        help="file path to load the discriminator",
    )
    parser.add_argument(
        "--num-attributes",
        default=1,
        type=str,
        help="number of attributes in the dataset",
    )
    parser.add_argument(
        "--use-CPU",
        default=True,
        type=bool,
        help="Specify whether you are loading trained models on a CPU",
    )
    args = parser.parse_args()
    return args


def build_test_data_loader(args):
    test_dataset = Datasets(
        root_images=args.root_rezimages,
        root_attributes=args.root_attributes,
        attributes=args.attr_chg,
        chunk="test",
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
    )


def load_networks(arg):
    encoder = Encoder()
    if arg.use_CPU:
        encoder.load_state_dict(
            torch.load(arg.encoder_fpath, map_location=torch.device("cpu"))
        )
    else:
        encoder.load_state_dict(torch.load(arg.encoder_fpath))

    decoder = Decoder(arg.num_attributes)
    if arg.use_CPU:
        decoder.load_state_dict(
            torch.load(arg.decoder_fpath, map_location=torch.device("cpu"))
        )
    else:
        decoder.load_state_dict(torch.load(arg.decoder_fpath))

    discriminator = Discriminator(arg.num_attributes)
    if arg.use_CPU:
        discriminator.load_state_dict(
            torch.load(arg.discriminator_fpath, map_location=torch.device("cpu"))
        )
    else:
        discriminator.load_state_dict(torch.load(arg.discriminator_fpath))

    return encoder, decoder, discriminator


def tensor_to_image(tensor):
    numpy_array = tensor.detach().cpu().numpy()
    return PIL.Image.fromarray(
        (
            255
            * (numpy_array.squeeze().transpose(1, 2, 0) - numpy_array.min())
            / (numpy_array.max() - numpy_array.min())
        ).astype(np.uint8)
    )


def grid_one_epoch(arg, encoder, decoder, discriminator, test_data_loader):
    encoder.eval()
    decoder.eval()
    discriminator.eval()

    nb_grid = 1

    for batch_idx, (batch_x, batch_y) in enumerate(
        tqdm(test_data_loader, desc="Test progress")
    ):
        if batch_idx % arg.sample_batch == 0:
            if arg.use_cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            y2flip = np.random.choice(
                arg.attr_chg, arg.nb_attributes2flip, replace=False
            )
            y2flip_idx = np.where(np.isin(arg.attr_chg, y2flip))[0]

            x2flip_idx = np.random.choice(
                batch_x.shape[0], arg.nb_x2flip, replace=False
            )
            print(len(x2flip_idx))
            x2flip = batch_x[x2flip_idx]

            Ex = encoder(batch_x)
            Ex2flip = Ex[x2flip_idx]

            y_true = batch_y[:, 0]
            y_true = y_true[x2flip_idx]
            y_flipped = 1 - y_true

            fig, axes = plt.subplots(
                arg.nb_attributes2flip * arg.nb_x2flip,
                arg.nb_alpha + 1,
                figsize=(18, 2 * arg.nb_x2flip),
            )

            idx = 0
            for x, z, y_flip in zip(x2flip, Ex2flip, y_flipped):
                z = torch.unsqueeze(z, dim=0)

                rect = patches.Rectangle(
                    (1, 1),
                    256,
                    256,
                    linewidth=4,
                    edgecolor="lightgreen",
                    facecolor="none",
                )
                if arg.nb_x2flip > 1:
                    axes[idx][0].imshow(tensor_to_image(x))
                    axes[idx][0].add_patch(rect)
                    axes[idx][0].axis("off")
                else:
                    axes[0].imshow(tensor_to_image(x))
                    axes[0].add_patch(rect)
                    axes[0].axis("off")

                title = (
                    arg.attr_chg[y2flip_idx[0]]
                    + ": "
                    + str(1 - y_flip[0].item())
                    + " > "
                    + str(y_flip[0].item())
                )
                alphas = (
                    np.linspace(0, 1, arg.nb_alpha)
                    if y_flip[0].item() == 1
                    else np.linspace(1, 0, arg.nb_alpha)
                )
                for idy, alpha in enumerate(alphas):
                    y_alpha = torch.tensor([alpha], dtype=torch.float32)

                    flipped_x = decoder(z, y_alpha)

                    if arg.nb_x2flip > 1:
                        axes[idx][idy + 1].imshow(
                            tensor_to_image(flipped_x)
                        )  # idy+1 , if : axes[idx][0].imshow(tensor_to_image(x))
                        axes[idx][idy + 1].axis("off")
                    else:
                        axes[idy + 1].imshow(tensor_to_image(flipped_x))
                        axes[idy + 1].axis("off")

                idx += 1

            plt.subplots_adjust(wspace=0, hspace=0)

            # Save the figure
            plt.savefig(
                f".\Results\grid\{arg.attr_chg[y2flip_idx[0]]}\{arg.attr_chg[y2flip_idx[0]]}_{nb_grid}.png"
            )
            nb_grid += 1
            plt.show()


def test_model(args, use_cuda=True):
    encoder, decoder, discriminator = load_networks(args)
    test_data_loader = build_test_data_loader(args)
    grid_one_epoch(args, encoder, decoder, discriminator, test_data_loader)


if __name__ == "__main__":
    args = parse_args()
    test_model(args)

# Autor Younes Nait_Omar

import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
description="Preprocess raw images and attributs to uniformed target."
    )
parser.add_argument(
        "--root-images",
        help="Directory path to raw images.",
        default="./Dataset/img_align_celeba/img_align_celeba",
        type=str,
    )
parser.add_argument(
        "--root-attributes",
        help="Directory path to raw attributes.",
        default="./Dataset/list_attr_celeba.csv",
        type=str,
    )
parser.add_argument(
        "--target-size",
        help="Target size to resize ",
        default=256,
        type=int,
    )
parser.add_argument(
        "--nb-images", help="number of images files extension to process.", default=202599, type=int)

args = parser.parse_args()

class PreProcessData:

    def __init__(self,args):
        self.number_images = args.nb_images
        self.images_size = args.target_size
        self.images_path = args.root_images 
        self.attributes_path = args.root_attributes

    def preprocess_images(self):
        print("Reading and Resizing images from img_align_celeba, wait few minutes ...")
        resized_images = []
        
        for i in tqdm(range(1, self.number_images + 1), desc="Redimensionnement"):
            img = cv2.imread(self.images_path + '/%06i.jpg' % i)
            img = img[20:-20]
            assert img.shape == (178,178,3)
            img =  cv2.resize(img, (self.images_size, self.images_size))
            resized_images.append(img)

        if len(resized_images) != self.number_images:
            raise Exception("Found %i images. Expected %i" % (len(resized_images), self.number_images))

        print('Enregistrement des données')
        if not os.path.exists('./Dataset/img_align_celeba_Preprocessed'):
            os.makedirs('./Dataset/img_align_celeba_Preprocessed')

        for i in tqdm(range(len(resized_images)), desc="Enregistrement"):
            assert resized_images[i].shape == (256,256,3)
            cv2.imwrite('./Dataset/img_align_celeba_Preprocessed/%06i.jpg'%(i+1),resized_images[i])

        print('Done !')


    
    def preprocess_attributes(self):
        df = pd.read_csv(self.attributes_path, index_col=0)

        attributes = {col: np.zeros(self.number_images, dtype=bool) for col in df.columns}

        for col in tqdm(df.columns, desc="Traitement des attributs"):
            attributes[col] = df[col].apply(lambda x: x == 1).to_numpy()

        output_path = './Dataset/processed_attributes'
        torch.save(attributes, output_path)
        print("Preprocessing completed!")




if __name__ == "__main__":
    preprocessor = PreProcessData(args)
    #preprocessor.preprocess_images()
    preprocessor.preprocess_attributes()

# ou lance directement la commande dans le terminal << python PreProcess.py --root-images "./Dataset/mg_align_celeba" --root-attributes "./Dataset/list_attr_celeba.csv" >>



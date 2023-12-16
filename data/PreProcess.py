# Import des bibliothèques nécessaires
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Preprocess raw images and attributs to uniformed target.")
parser.add_argument("--root-images", help="Directory path to raw images.", default="./archive/img_align_celeba/img_align_celeba", type=str)
parser.add_argument("--root-attributes", help="Directory path to raw attributes.", default="./archive/list_attr_celeba.csv", type=str)
parser.add_argument("--target-size", help="Target size to resize ", default=256, type=int)
parser.add_argument("--nb-images", help="number of images files extension to process.", default=202599, type=int)

args = parser.parse_args()

class PreProcessData:
    def __init__(self, args):
        self.number_images = args.nb_images # Nm d'images à traiter
        self.images_size = args.target_size # La target size
        self.images_path = args.root_images #  Chemin des images
        self.attributes_path = args.root_attributes #  Chemin des attribues

        # Vérification des chemins
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(f"Images path '{self.images_path}' does not exist.")
        if not os.path.exists(self.attributes_path):
            raise FileNotFoundError(f"Attributes path '{self.attributes_path}' does not exist.")

    def preprocess_images(self):
        """
        Lire et redimensionner les images brutes.
        """
        print("Reading and Resizing images from img_align_celeba, wait a few minutes...")
        resized_images = []

        for i in tqdm(range(1, self.number_images + 1), desc="Resizing"):
            img_path = os.path.join(self.images_path, f'{i:06}.jpg')
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise IOError(f"Failed to read image: {img_path}")

                img = img[20:-20] # Crop les images
                assert img.shape == (178, 178, 3), f"Image shape mismatch: {img.shape}"
                img = cv2.resize(img, (self.images_size, self.images_size))
                resized_images.append(img) # Ajout de l'image redimensionnée à la liste
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        if len(resized_images) != self.number_images:
            raise Exception(f"Found {len(resized_images)} images. Expected {self.number_images}")

        print('Saving resized images...')
        if not os.path.exists('resized_images'):
            os.makedirs('resized_images')
            
        # Enregistrement des images 
        for i, img in enumerate(tqdm(resized_images, desc="Saving")):
            assert img.shape == (256, 256, 3), "Resized image shape mismatch"
            cv2.imwrite(f'resized_images/{i+1:06}.jpg', img)

        print('Done!')

    def preprocess_attributes(self):
        """
        Prétraiter les attributs et les sauvegarder.
        """
        try:
            df = pd.read_csv(self.attributes_path, index_col=0)
        except Exception as e:
            raise Exception(f"Error reading attributes file: {e}")

        # Vérification
        assert not df.empty, "Attributes DataFrame is empty"
        assert all(isinstance(col, str) for col in df.columns), "Column names should be strings"

        attributes = {col: np.zeros(self.number_images, dtype=bool) for col in df.columns}

        # Conversion des attributs
        for col in tqdm(df.columns, desc="Processing attributes"):
            attributes[col] = df[col].apply(lambda x: x == 1).to_numpy()

        output_path = 'processed_attributes'
        torch.save(attributes, output_path) # Sauvegarder
        print("Preprocessing completed!")



if __name__ == "__main__":
    preprocessor = PreProcessData(args)
    preprocessor.preprocess_images() # Appel à la fonction pour le traitement des images
    preprocessor.preprocess_attributes() # Appel à la fonction pour le traitement des attributs
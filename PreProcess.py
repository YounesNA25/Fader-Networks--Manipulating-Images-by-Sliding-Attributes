import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd



class PreProcessData:

    def __init__(self, images_size=256, number_images=202599, images_path='archive/img_align_celeba/img_align_celeba', attributes_path='archive/list_attr_celeba.csv'):
        self.number_images = number_images
        self.images_size = images_size
        self.images_path = images_path 
        self.attributes_path = attributes_path

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
        if not os.path.exists('resized_images'):
            os.makedirs('resized_images')

        for i in tqdm(range(len(resized_images)), desc="Enregistrement"):
            assert resized_images[i].shape == (256,256,3)
            cv2.imwrite('resized_images/%06i.jpg'%(i+1),resized_images[i])

        print('Done !')


    
    def preprocess_attributes(self):
        df = pd.read_csv(self.attributes_path, index_col=0)

        attributes = {col: np.zeros(self.number_images, dtype=bool) for col in df.columns}

        for col in tqdm(df.columns, desc="Traitement des attributs"):
            attributes[col] = df[col].apply(lambda x: x == 1).to_numpy()

        output_path = 'processed_attributes'
        torch.save(attributes, output_path)
        print("Preprocessing completed!")






# Example usage:
if __name__ == "__main__":
    preprocessor = PreProcessData()
    #preprocessor.preprocess_images()
    preprocessor.preprocess_attributes()



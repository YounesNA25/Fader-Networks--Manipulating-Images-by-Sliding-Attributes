import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import csv


class PreProcessData:

    def __init__(self, images_size=256, number_images=202599, images_path='archive/img_align_celeba/img_align_celeba', attributes_path='archive/list_attr_celeba.csv'):
        self.number_images = number_images
        self.images_size = images_size
        self.images_path = images_path 
        self.attributes_path = attributes_path

    def preprocess_images(self):


        print("Reading and Resizing images from img_align_celeba, wait few minutes ...")
        resized_images = []
        for i in range(1, self.number_images + 1):
            img = cv2.imread(self.images_path + '/%06i.jpg' % i)
            img = img[20:-20]
            assert img.shape == (178,178,3)
            img =  cv2.resize(img, (self.images_size, self.images_size))
            resized_images.append(img)
        
        if len(resized_images) != self.number_images:
            raise Exception("Found %i images. Expected %i" % (len(resized_images), self.number_images))
        
        print('enregistrement des données')
        if not os.path.exists('resized_images'):
            os.makedirs('resized_images')

        for i, image in enumerate(resized_images):
            assert resized_images[i].shape == (256,256,3)
            cv2.imwrite('resized_images/%06i.jpg'%(i+1),resized_images[i])
            
        print('done ! ...')


    def preprocess_attributes(self):

        
        with open(self.attributes_path, 'r') as file_path :
            attributes_lines = [line.rstrip() for line in file_path.readlines()]

        attribute_keys = attributes_lines[0].split(',')
        attribute_keys = attribute_keys[1:]
        attributes = {k: np.zeros(self.number_images, dtype=bool) for k in attribute_keys}
        for i, line in enumerate(attributes_lines[1:]):
            lines = line.split(',')
          
            for j, value in enumerate(lines[1:]):
                attributes[attribute_keys[j]][i] = value == '1'
        
        print(attributes[attribute_keys[0]][5])

        # Vérification si le répertoire existe
        repertoire = 'processed_attributes.txt'
        
        # Ouverture du fichier en mode écriture dans le répertoire spécifié
        with open(f'{repertoire}', 'w') as f:
            for i in range(1,len(attribute_keys)):
                f.write(f'{attribute_keys[i]} : ')
                for j in range(self.number_images):
                    f.write(f'{attributes[attribute_keys[i]][j]} ')
                f.write(f'\n')
        
        print("done ! ")


    


# Example usage:
if __name__ == "__main__":
    preprocessor = PreProcessData()
    preprocessor.preprocess_images()
    preprocessor.preprocess_attributes()



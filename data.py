import os
from glob import glob
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms


class Datasets(torch.utils.data.Dataset):
    def __init__(self, root_images, root_attributes, attributes='Male', ext="jpg", chunk=None):
        self.files = glob(os.path.join(root_images, f"*.{ext}"))
        self.attributes = torch.load(root_attributes)
        self.attr_labels = torch.stack([torch.tensor(self.attributes[attr]) for attr in attributes], dim=1)
        self.transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5] , std=[0.5, 0.5, 0.5]) ])  # [0, 255] --> [ 0., 1.]
        
        
        # Sélection des indices en fonction du chunk
        if chunk == 'train':
            self.indices = torch.arange(0, 162770)
        elif chunk == 'test':
            self.indices = torch.arange(162770, 182637)
        elif chunk == 'val':
            self.indices = torch.arange(182637, 202599)
        else:
            self.indices = torch.arange(0, len(self.files))  # Utiliser tous les fichiers si aucun chunk spécifié

        # Filtrer les fichiers et les attributs selon les indices sélectionnés
        self.files = [self.files[i] for i in self.indices]
        self.attr_label = self.attr_labels[self.indices]

    @staticmethod
    def mapper(value):
        return [0, 1] if value == 0. else [1, 0]

    def __getitem__(self, index):
        file_path = self.files[index]
        image = Image.open(file_path)
        image = self.transform(image)
        attributes_onehot = [Datasets.mapper(val) for val in self.attr_labels[index]]
        attributes = [1 if val else 0 for val in self.attr_labels[index]]

        return image, attributes_onehot

    def __len__(self):
        return len(self.files)
    

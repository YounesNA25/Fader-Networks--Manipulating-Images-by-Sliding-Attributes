import os
from glob import glob
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms


class Datasets(torch.utils.data.Dataset):
    def __init__(self, root_images, root_attributes, attributes='Male', ext="jpg", chunk=None):
        self.files = sorted(glob(os.path.join(root_images, f"*.{ext}")))
        self.attributes = torch.load(root_attributes)
        self.attr_labels = torch.stack([torch.tensor(self.attributes[attr]) for attr in attributes], dim=1)
        self.transform = transforms.Compose([
        transforms.ToTensor()])
        
        # Sélection des indices en fonction du chunk
        if chunk == 'train':
            # self.indices = torch.arange(0, 162770)
            self.indices = torch.arange(0,1000)
        elif chunk == 'test':
            # self.indices = torch.arange(162770, 182637)
            self.indices = torch.arange(1000,1500)
        elif chunk == 'val':
            # self.indices = torch.arange(182637, 202599)
            self.indices = torch.arange(1500,2000)
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
        attributes = [Datasets.mapper(val) for val in self.attr_labels[index]]

        return image, torch.tensor(attributes)[0,:]

    def __len__(self):
        return len(self.files)


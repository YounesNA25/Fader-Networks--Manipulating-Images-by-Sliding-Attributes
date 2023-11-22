import os
from glob import glob
import torch
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd




from PIL import Image

class Datasets(torch.utils.data.Dataset):
    def __init__(self, root_images,root_atributes, input_size = 256, ext = "jpg"):
        self.files = glob(os.path.join(root_images, f"*.{ext}"))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((178, 178)),
            transforms.Resize((input_size,input_size))
            ])
        self.atributes = pd.read_csv(root_atributes, usecols=lambda column: column not in ['image_id'])
        

    def __getitem__(self, index):
        file_path = self.files[index]
        image = Image.open(file_path)
        image = self.transform(image)
        image_attributes = self.atributes.iloc[index].values 
        image_attributes = [0 if value == -1 else 1 for value in image_attributes.tolist()]
        tensor_attributes = torch.tensor(image_attributes, dtype=torch.float32)

        return image, tensor_attributes

    def __len__(self):
        return len(self.files)
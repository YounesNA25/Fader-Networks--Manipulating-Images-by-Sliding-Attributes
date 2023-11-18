import os
from glob import glob
import torch
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd




from PIL import Image

class Datasets(torch.utils.data.Dataset):
<<<<<<< Updated upstream
    def __init__(self, root_images,root_atributes, input_size = 178, ext = "jpg"):
=======
    def __init__(self, root_images,root_atributes, input_size = 256, ext = "jpg"):
>>>>>>> Stashed changes
        self.files = glob(os.path.join(root_images, f"*.{ext}"))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size,input_size))
            ])
<<<<<<< Updated upstream
        self.atributes = pd.read_csv("archive/list_attr_celeba.csv", usecols=lambda column: column not in ['image_id'])
=======
        self.atributes = pd.read_csv(root_atributes, usecols=lambda column: column not in ['image_id'])
>>>>>>> Stashed changes
        

    def __getitem__(self, index):
        file_path = self.files[index]
        image = Image.open(file_path)
        image = self.transform(image)
        image_attributes = self.atributes.iloc[index].values
        tensor_attributes = torch.tensor(image_attributes, dtype=torch.float32)

        return image, tensor_attributes

    def __len__(self):
        return len(self.files)
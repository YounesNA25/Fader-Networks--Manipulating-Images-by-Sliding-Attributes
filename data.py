import os
from glob import glob
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

class Datasets(torch.utils.data.Dataset):
    def __init__(self, root, input_size = 178, ext = "jpg"):
        self.files = glob(os.path.join(root, f"*.{ext}"))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((178, 178)),
            transforms.Resize((input_size,input_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):
        file_path = self.files[index]
        image = Image.open(file_path)
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.files)
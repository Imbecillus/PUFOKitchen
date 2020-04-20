import torch
import torchvision
import imageio
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def load_image(path):
    image = imageio.imread(path)
    image = Image.fromarray(image)
    return image

class PUFOKitchen(Dataset):
    def __init__(self, root, data_transforms=None):
        # INIT
        super(PUFOKitchen, self).__init__()

        if data_transforms is None:
            data_transforms = []

        self.root = root
        self.data_transforms = data_transforms

        # GENERATE LIST OF ALL IMAGE FILES WITH THEIR RESPECTIVE CLASSES
        self.image_list = []
        classes = [o for o in os.listdir(root)
                   if os.path.isdir(os.path.join(root, o))]

        for c in classes:
            class_directory = os.path.join(root, c)
            files = [f for f in os.listdir(class_directory)
                     if f.endswith('jpg')]
            for f in files:
                filepath = os.path.join(root, c, f)
                class_number = classes.index(c)
                entry = (filepath, class_number)
                self.image_list.append(entry)

    def __getitem__(self, number):
        path, c = self.image_list[number]

        image = imageio.imread(path)                    # Load image
        image = Image.fromarray(image)                  # Convert to PIL format
        image = self.data_transforms(image)             # Apply transforms
        image = np.array(image)                         # Convert to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float) 
        image = image / 255
        image = image.permute(2, 0, 1)                  # Reshape the tensor to [C,H,W]-format

        return (image, c)

    def __len__(self):
        return len(self.image_list)
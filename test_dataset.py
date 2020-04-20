# DEBUG
# Some variables that will later be given as command line parameters are set here
trainset_path = "alldata"
testset_path = "alldata"

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

transforms = [torchvision.transforms.Resize((256, 256))]

classes = [o for o in os.listdir(trainset_path)
           if os.path.isdir(os.path.join(trainset_path, o))]
print(f'Our classes are: {classes}')

from dataset import PUFOKitchen
dataset = PUFOKitchen(trainset_path, data_transforms=torchvision.transforms.Compose(transforms))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

dataiter = iter(dataloader)

images, labels = dataiter.next()

# print images
for i in range(len(labels)):
    print(classes[labels[i]])
    plt.imshow(images[i].permute(1, 2, 0))
    plt.show()
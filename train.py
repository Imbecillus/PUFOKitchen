import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

transforms = [torchvision.transforms.Resize((256, 256))]

classes = [o for o in os.listdir('data')
           if os.path.isdir(os.path.join('.', 'data', o))]
print(f'Our classes are: {classes}')

from dataset import PUFOKitchen
dataset = PUFOKitchen('data', data_transforms=torchvision.transforms.Compose(transforms))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

dataiter = iter(dataloader)

images, labels = dataiter.next()

# print images
for i in range(len(labels)):
    print(classes[labels[i]])
    plt.imshow(images[i].permute(1, 2, 0))
    plt.show()
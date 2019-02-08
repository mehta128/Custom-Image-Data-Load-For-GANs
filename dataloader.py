import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

#Torchvision to load dataset, and transform the dataset according to your requirement.
data_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def load_dataset():
    data_path = 'data/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=data_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=0,
        shuffle=True
    )
    return train_loader


#Use iterator to iterate over the datasets
train_iter = iter(load_dataset())

#Once you have your data in form of tensors, train your network or do your desired goal.

print(type(train_iter))

images, labels = train_iter.next()
print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))


# The generated images from GANs will be in tensors, transpose them to plot it on graph
grid = torchvision.utils.make_grid(images)
plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.show()

grid = torchvision.utils.make_grid(images)
# plt.imshow((grid * 255).astype(np.uint8))
plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.show()

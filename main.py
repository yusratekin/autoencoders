"""
problem tanimi : veri sikistitmasi -> autoencoders
veri : FashionMINST

"""
# import
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np


# %% veri seti yukleme ve on isleme

transform = transforms.Compose([transforms.ToTensor()])  # goruntuyu tensore cevir

# egitim ve test veri setini indir ve yukle
train_dataset = datasets.FashionMNIST(root = '.data', train = True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root = '.data', train = False, transform=transform, download=True)


# batch size
batch_size = 64


# egitim ve test veri yukleyicileri olusturma
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
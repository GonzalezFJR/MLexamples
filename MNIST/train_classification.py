import torch
import torch.nn as nn
import numpy as np
import os
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import pickle as pkl
from MLtools.dataloader_tools import  CreateDataloadersDict, GetDatasetSizes

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist_train = torchvision.datasets.MNIST('', train=True, transform =transform, download=True)

# Test, validation and training sets
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [int(np.floor(len(mnist_train)*0.75)), int(np.ceil(len(mnist_train)*0.25))])
mnist_test = torchvision.datasets.MNIST('', train=False, transform = transform, download=True)

dataloaders = CreateDataloadersDict(mnist_train, mnist_test, mnist_val, 100)
dataset_sizes = GetDatasetSizes(dataloaders)
print(f'Dataset size: {dataset_sizes}')

### Load the model
from models import CNNClassifier
learning_rate = 0.001
num_epochs = 10

model = CNNClassifier().to(device)
path = './models/CNNclassifier.model'
trainIfSaved = False
print(model)

if __name__ == '__main__':
  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

  ### Train and save
  from MLtools.train_tools import train_model
  if not os.path.isfile(path) or trainIfSaved:
    model, training_curves = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device, save=path, supervised=True)
  else:
    model = torch.load(path)
    model.eval()
    training_curves = pkl.load(path+'.curves')

import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.flatten import Flatten

''' Autoencoder CNN '''
class CNNAutoEncoder(nn.Module):
    # 28x28
    def __init__(self):
        super(CNNAutoEncoder, self).__init__()
        # Dividir el codificador y el decodificador

        self.encoder = nn.Sequential(
            # Realizamos una convolución del input
            # in_channels = 1 porque el input es una imagen en escala de grises
            # El tamaño de la capa Conv2d da lugar a que el tamaño del output se modifique a (8, 28, 28). 
            nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # dimensiones a (8, 14, 14)
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(8*14*14, 32), # codificación de tamaño 32
        )

        self.decoder = nn.Sequential(
            # El input del decodificador será un tensor lineal de longitud 32
            nn.Linear(32, 8*14*14),
            nn.ReLU(),
            nn.Unflatten(dim = 1, unflattened_size = (8, 14, 14)),
            nn.ReLU(),
            # ConvTranspose2d es una deconvolución que devuelve a las dimensiones originales (1, 28, 28)
            nn.ConvTranspose2d(in_channels = 8, out_channels = 1, kernel_size = 2, stride = 2),
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def encode(self, x):
        return self.encoder(x)

''' CNN para clasificar en 10 categorías '''
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.pipeline = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Linear(8*14*14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        return self.pipeline(x)

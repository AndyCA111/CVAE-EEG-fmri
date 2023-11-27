import numpy as np
import os
import shutil
import sys
import torch
# import tensorflow as tf
from models.cvae import CVAE
from torch.nn import functional as F
from torchvision import datasets, transforms

def one_hot(labels, class_size, device):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)

def loss_function(recon_x, x, mu, logvar):
        #kl-divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #reconstruction
        REC = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        return KLD + REC
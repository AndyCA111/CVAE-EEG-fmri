import argparse
import numpy as np
import os
# import tensorflow as tf
import torch
#from codebase import utils as ut
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image 
import models.utils as ut

path_to_save ='/Users/binxuli/Documents/cs236/proj/eeg2fmri/CVAE-EEG-fmri/result'

def train(model, epoch, train_loader, device, optimizer, class_num):
    model.train()
    train_loss = 0 
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        label = ut.one_hot(label, class_num, device)
        recon_batch, mu, logvar = model(data, label)
        optimizer.zero_grad()
        loss = ut.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy() # use detach to divide loss and grad, then convert to numpy, only can use cpu
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(model, epoch, test_loader, device, class_num):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            labels = ut.one_hot(labels, class_num, device)
            recon_batch, mu, logvar = model(data, labels)
            test_loss += ut.loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
            if i == 0:
                n = min(data.size(0), 5)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), path_to_save +
                         '/reconstruction_' + str(f"{epoch:02}") + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


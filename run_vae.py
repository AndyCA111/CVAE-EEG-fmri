import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models.train import train
from models.train import test
from models.cvae import CVAE
#setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#kwargs = {'num_workers': 1, 'pin_memory': True}  using gpu
#hyper parameter
batch_size = 64 
latent_size = 20
epochs = 10
# class
num_class = 10
# input size
inputsize = 784 #28*28
path_to_save ='/Users/binxuli/Documents/cs236/proj/eeg2fmri/CVAE-EEG-fmri/result'
#load train and test
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False)




############## bulid model #####################################



model = CVAE(inputsize, latent_size, num_class).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
        train(model, epoch, train_loader, device, optimizer, num_class)
        test(model, epoch, test_loader, device, num_class)
        with torch.no_grad():
            c = torch.eye(10, 10).to(device)
            sample = torch.randn(10, latent_size).to(device)
            sample = model.decode(sample, c).cpu()
            save_image(sample.view(10, 1, 28, 28), path_to_save +
                       '/sample_' + str(f"{epoch:02}") + '.png')
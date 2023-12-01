import torch
from torch import nn
from torch.nn import functional as F

class CVAE3D(nn.Module):
    def __init__(self, feature_size=256, latent_size=128, conditional_size=256, hidden_size=400):
        super(CVAE3D, self).__init__()
        self.feature_size = feature_size
        self.conditional_size = conditional_size

        # encode
        # input size: batch, 1, 16, 64, 64
        self.conv1  = nn.Conv3d(1, 32, kernel_size=3, stride=(2, 2, 2), padding=1) # , 8, 32, 32
        self.enc_bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1) # 4, 16, 16
        self.enc_bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(64, 32, kernel_size=3, stride=(1, 2, 2), padding=1) #4, 8, 8
        self.enc_bn3 = nn.BatchNorm3d(8)
        self.conv4 = nn.Conv3d(32, 1, kernel_size=3, stride=(1, 2, 2), padding=1) # 4, 4, 4
        self.flat = nn.Flatten() 
        # ---- End of encoder feature extraction -----

        # fc to z-dim
        self.fc11 = nn.Linear(64 + conditional_size, latent_size)
        self.fc12 = nn.Linear(64 + conditional_size, latent_size)

        # decoder, conv3d + upsample
        # latent to a linear [4, 4, 4]
        self.d_fc = nn.Linear(latent_size + conditional_size, 4*4*4*8)
        self.tp_conv1 = nn.Conv3d(8, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec_bn1 = nn.BatchNorm3d(32)
        self.tp_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec_bn2 = nn.BatchNorm3d(64)
        self.tp_conv3 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec_bn3 = nn.BatchNorm3d(32)
        self.tp_conv4 = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, condition): # Q(z|x, c) condition:(batch, conditional_size); x: (batch, 104, 106, 16)
        '''
        x dimension: (batch_size, channel, depth, height, width)
        c dimension: (batch_size, feature_size)
        '''
        # inputs = torch.cat([x, condition], 1) # (bs, feature_size+class_size)
        # h1 = self.elu(self.fc1(inputs))
        x1 = F.leaky_relu(self.enc_bn1(self.conv1(x)))
        print(x1.shape)
        x2 = F.leaky_relu(self.enc_bn2(self.conv2(x1)))
        print(x2.shape)
        x3 = F.leaky_relu(self.enc_bn3(self.conv3(x2)))
        x4 = self.conv4(x3)
        flat = self.flat(x4)
        print(flat.shape)
        conditioned = torch.concat([flat, condition], dim=-1)
        z_mu = self.fc11(conditioned)
        z_var = self.fc12(conditioned)
        return z_mu, z_var

    def gaussian_sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        conditioned_noise = torch.cat([z, c], 1) # (bs, latent_size+conditional_size)
        res = self.d_fc(conditioned_noise)
        res = torch.reshape(res, (-1, 8, 4, 4, 4))
        res = nn.ReLU()(self.dec_bn1(self.tp_conv1(res))) # batch, 32, 4, 4, 4
        print(res.shape)
        res = F.interpolate(res, scale_factor=2) # batch, 32, 8, 8, 8
        print(res.shape)
        assert res.shape[1:] == (32, 8, 8, 8)

        res = nn.ReLU()(self.dec_bn2(self.tp_conv2(res))) # batch, 64, 8, 8, 8
        res = F.interpolate(res, scale_factor=2) # batch, 64, 16, 16, 16
        assert res.shape[1:] == (64, 16, 16, 16)

        res = nn.ReLU()(self.dec_bn3(self.tp_conv3(res))) # batch, 32, 16, 16, 16
        res = F.interpolate(res, scale_factor=(1, 2, 2)) # batch, 32, 16, 32, 32
        assert res.shape[1:] == (32, 16, 32, 32)

        res = nn.ReLU()(self.tp_conv4(res)) # batch, 1, 16, 32, 32
        res = F.interpolate(res, scale_factor=(1, 2, 2)) # batch, 1, 16, 64, 64
        assert res.shape[1:] == (1, 16, 64, 64)

        return res

    def forward(self, x, c):
        mu, logvar = self.encode(x, c) 
        z = self.gaussian_sample(mu, logvar)
        return self.decode(z, c), mu, logvar


if __name__ == "__main__":
    input = torch.zeros((1, 1, 16, 64, 64))
    c = torch.ones((1, 256))
    model = CVAE3D()
    decoded, mu, logvar = model(input, c)
    print(decoded.shape)
    assert decoded.shape == input.shape
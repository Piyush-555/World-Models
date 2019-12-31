import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 32, 4, 2),
                        nn.ReLU()
        )

        self.conv2 = nn.Sequential(
                        nn.Conv2d(32, 64, 4, 2),
                        nn.ReLU()
        )

        self.conv3 = nn.Sequential(
                        nn.Conv2d(64, 128, 4, 2),
                        nn.ReLU()
        )

        self.conv4 = nn.Sequential(
                        nn.Conv2d(128, 256, 4, 2),
                        nn.ReLU()
        )

        self.mu = nn.Sequential(
                        nn.Linear(2*2*256, 32)
        )

        self.logvar = nn.Sequential(
                        nn.Linear(2*2*256, 32)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 2*2*256)

        z_mu = self.mu(x)
        z_logvar = self.logvar(x)

        # sampling
        std = torch.exp(z_logvar / 2)
        eps = torch.randn_like(std)
        latent_z = eps*std + z_mu

        return latent_z


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.dense = nn.Sequential(
                        nn.Linear(32, 1024)
        )

        self.deconv1 = nn.Sequential(
                        nn.ConvTranspose2d(1024, 128, 5, 2),
                        nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 5, 2),
                        nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
                        nn.ConvTranspose2d(64, 32, 6, 2),
                        nn.ReLU()
        )

        self.deconv4 = nn.Sequential(
                        nn.ConvTranspose2d(32, 3, 6, 2)
        )

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, 1024, 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return torch.sigmoid(x)


class VAE(nn.Module):

    def __init__(self, enc, dec):
        super().__init__()

        self.encoder = enc
        self.decoder = dec

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed


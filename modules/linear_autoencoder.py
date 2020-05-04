import torch.nn as nn
import torch

from autoencoder import Encoder, Decoder

class LinearEncoder(Encoder):


    def __init__(self, args, device, logger = None):
        super().__init__(args, device, logger)

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(True),

            nn.Linear(self.latent_dim * 2, self.latent_dim * 4),
            nn.ReLU(True),

            nn.Linear(self.latent_dim * 4, self.dim_z)
        ).to(self.device)


    def forward(self, x):
        return super().forward(x)


class LinearDecoder(Decoder):


    def __init__(self, args, device, logger = None):
        super().__init__(args, device, logger)

        self.model = nn.Sequential(
            nn.Linear(self.dim_z, self.dim_z * 2),
            nn.ReLU(True),

            nn.Linear(self.dim_z * 2, self.dim_z * 4),
            nn.ReLU(True),

            nn.Linear(self.dim_z * 4, self.latent_dim)
        ).to(self.device)

        self.model = self.model.to(self.device)


    def forward(self, z):
        return super().forward(z)
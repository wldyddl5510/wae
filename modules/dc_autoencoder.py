from autoencoder import Encoder, Decoder
import torch.nn as nn

"""
    Deep Convolution Encoder
"""
class DcEncoder(Encoder):


    def __init__(self, args, device, logger = None):
        super().__init__(args, device, logger)

        # Deep convolution encoder
        self.model = nn.Sequential(
            # from input to latent
            nn.Conv2d(in_channels = args.n_channels, out_channels = self.latent_dim, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.ReLU(True),

            # latent double
            nn.Conv2d(self.latent_dim, self.latent_dim * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.latent_dim * 2),
            nn.ReLU(True),

            # latent **2
            nn.Conv2d(self.latent_dim * 2, self.latent_dim * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.latent_dim * 4),
            nn.ReLU(True),

            # latent ** 3
            nn.Conv2d(self.latent_dim * 4, self.latent_dim * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.latent_dim * 8),
            nn.ReLU(True),
        ).to(self.device)

        # project to z space
        self.final_layer = nn.Linear(self.latent_dim * 8, self.dim_z).to(self.device)

    def forward(self, image):
        z = self.model(image)
        z = z.squeeze()
        z = self.final_layer(z)
        return z

"""
    Dc Decoder class
"""
class DcDecoder(Decoder):


    def __init__(self, args, device, logger = None):
        super().__init__(args, device, logger)

        # from latent to cnn
        self.project = nn.Sequential(
            nn.Linear(self.dim_z, self.latent_dim * 8 * 7 * 7),
            nn.ReLU(True)
        ).to(self.device)

        # cnn decoder
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim * 8, self.latent_dim * 4, 4),
            nn.BatchNorm2d(self.latent_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.latent_dim * 4, self.latent_dim * 2, 4),
            nn.BatchNorm2d(self.latent_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.latent_dim * 2, 1, 4, stride = 2),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, z):
        z = self.project(z)
        z = z.view(-1, self.latent_dim * 8, 7, 7)
        reconst_x = self.model(z)
        return reconst_x
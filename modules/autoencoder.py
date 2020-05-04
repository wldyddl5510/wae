from abc import abstractmethod, ABC
import torch.nn as nn

"""
    Abstract class for autoencoder

    Have Encoder class and Decoder class
"""

"""
    Abstract Encoder
"""
class Encoder(ABC, nn.Module):
    

    def __init__(self, args, device, logger = None):
        super().__init__()
        # Actual Neural Network
        self.model = None
        self.final_layer = None
        # Exist loggers?
        self.logger = logger
        # Latent dim = input dim
        self.latent_dim = args.latent_dim
        # Z space dim
        self.dim_z = args.dim_z
        # GPU or CPU
        self.device = device
        
    def forward(self, x):
        # Pass through the encoder: X -> Z
        z = self.model(x)
        # Return Z space
        return z

    # Network summary
    def summary(self):
        if self.logger is not None:
            net_parameters = filter(lambda p: p.requires_grad, self.parameters())
            params = sum([np.prod(p.size()) for p in net_parameters])
            self.logger.info('Trainable parameters: {}'.format(params))
            self.logger.info(self)
        else:
            pass


"""
    Abstract Decoder
"""

class Decoder(ABC, nn.Module):


    def __init__(self, args, device, logger = None):
        super().__init__()
        # actual neural network
        self.project = None
        self.model = None
        # Input dim
        self.latent_dim = args.latent_dim
        # Z space dim
        self.dim_z = args.dim_z
        # Cpu or Gpu
        self.device = device
        
    def forward(self, z):
        # pass the decoder: Z -> X_tilda
        x_tilda = self.model(z)
        # return reconstructed space
        return x_tilda

    # Network summary
    def summary(self):
        if self.logger is not None:
            net_parameters = filter(lambda p: p.requires_grad, self.parameters())
            params = sum([np.prod(p.size()) for p in net_parameters])
            self.logger.info('Trainable parameters: {}'.format(params))
            self.logger.info(self)
        else:
            pass

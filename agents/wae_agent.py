from autoencoder_agent import AutoencoderAgent
from abc import ABC, abstractmethod

"""
    Agent for wasserstein autoencoder
"""

class WaeAgent(AutoencoderAgent):


    def __init__(self, args, module, env, device, logger = None):
        super().__init__(args, module, env, device, logger)
        self.prior = args.prior

    @abstractmethod
    def compute_wasser_ae(self, input_x, latent_z, reconst_x):
        pass

    def loss_ae(self, input_x, latent_z, reconst_x):
        return self.compute_wasser_ae(input_x, latent_z, reconst_x)
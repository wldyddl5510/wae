# Abstract class
from abc import abstractmethod, ABC
from modules.autoencoder import Encoder, Decoder

# Torch
import torch
import numpy as np
import torch.autograd as autograd

# saving
from pathlib import Path
import os.path

# Tensorboard
from tensorboardX import SummaryWriter

from utils import NORM
from paths import RESULT_PATH

import pdb


"""
    Abstract class for AutoEncoder Agent
"""

class AutoencoderAgent(ABC):

    
    def __init__(self, args, module, env, device, logger = None):
        # Modules
        self.encoder = module.encoder(args, device)
        self.decoder = module.decoder(args, device)

        # env
        self.env = env

        # agent
        self.agent = args.agent

        # training options
        self.epoches = args.epoches
        self.latent_dim = args.latent_dim
        self.n_critic = args.n_critic
        self.dim_z = args.dim_z

        # optimizers
        self.optim_E = torch.optim.Adam(self.encoder.parameters(), lr = args.lr)
        self.optim_D = torch.optim.Adam(self.decoder.parameters(), lr = args.lr)

        # scheduler
        self.scheduler_E = torch.optim.lr_scheduler.StepLR(self.optim_E, step_size = args.step_size, gamma = args.gamma)
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optim_D, step_size = args.step_size, gamma = args.gamma)

        # cpu? gpu?
        self.device = device

        # tensorboard?
        self.tensorboard = args.tensorboard
        self.sample_interval = args.sample_interval

        # logger?
        self.logger = logger

        self.norm = NORM[args.norm]
    
    @abstractmethod
    def loss_ae(self, input_x, latent_z, reconst_x):
        pass

    def reconstruction_loss(self, input_x, reconst_x):
        loss = torch.mean(torch.sum(((reconst_x - input_x.float()) ** self.reconstruct_norm), dim = 1))
        return loss.to(self.device)

    def train(self):
        # tensorboard logging
        if self.tensorboard:
            writer = SummaryWriter()

        dirname = os.path.join(RESULT_PATH, self.env.env_name, self.agent, type(self.encoder).__name__)
        Path(dirname).mkdir(parents = True, exist_ok = True)
        

        # Train data
        train_dataloader = self.env.train

        batches_done = 0

        self.encoder.train()
        self.decoder.train()

        # Loop epoches
        for epoch in range(self.epoches):

            # loop dataloader
            for i, (data, _) in enumerate(train_dataloader):
                # batch data
                input_data_x = data.to(self.device)

                # train encoder / decoder
                self.optim_E.zero_grad()
                self.optim_D.zero_grad()

                # network
                latent_z = self.encoder(input_data_x)
                reconst_x = self.decoder(latent_z)

                ae_loss = self.loss_ae(input_data_x, latent_z, reconst_x)
                # Optimization
                ae_loss.backward()

                self.optim_E.step()
                self.optim_D.step()

                input_reconst_loss = self.reconstruction_loss(input_data_x.data, reconst_x.data)

                if self.logger is not None:
                    self.logger.info("[Epoch %d/%d] [Batch %d/%d] [AE loss: %f] [input_reconst loss: %f]"
                            % (epoch, 
                            self.epoches, 
                            i, 
                            len(train_dataloader), 
                            ae_loss.data.item(), 
                            input_reconst_loss.data.item()))
                
                if self.tensorboard:
                    logging_info = {
                        'AE loss': ae_loss.data.item(),
                        'input_reconst loss': input_reconst_loss.data.item()
                    }

                    # graph
                    for tag, value in logging_info.items():
                        writer.add_scalar(tag, value, batches_done)
                    
                    # images
                    if batches_done % self.sample_interval == 0:
                        tag = self.env.env_name + " " + self.agent + " " + type(self.encoder).__name__ + " %d" %batches_done
                        writer.add_images(tag, reconst_x.view(self.env.batch_size, 1, 28, 28).data, batches_done)

                batches_done += self.n_critic

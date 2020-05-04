from autoencoder_agent import AutoencoderAgent
from abc import ABC, abstractmethod

# Torch
import torch
import numpy as np
import torch.autograd as autograd

# utils
from utils import ASSYMETRIC_DIVERGENCES

# saving
from pathlib import Path
import os.path

# Tensorboard
from tensorboardX import SummaryWriter

import pdb


class InfoAeAgent(AutoencoderAgent, ABC):


    def __init__(self, args, module, env, device, logger = None):
        super().__init__(args, module, env, device, logger)
        # Second en/decoder
        self.second_encoder = module.encoder(args, device)
        self.second_decoder = module.decoder(args, device)

        # Optimizers for second layer
        self.optim_second_E = torch.optim.Adam(self.second_encoder.parameters(), lr = args.lr)
        self.optim_second_D = torch.optim.Adam(self.second_decoder.parameters(), lr = args.lr)

        # scheduler 
        self.scheduler_second_E = torch.optim.lr_scheduler.StepLR(self.optim_second_E, step_size = args.step_size, gamma = args.gamma)
        self.scheduler_second_D = torch.optim.lr_scheduler.StepLR(self.optim_second_D, step_size = args.step_size, gamma = args.gamma)

        # weights 
        self.lambda_z_z_tilda = args.lambda_z_z_tilda
        self.lambda_input_reconst = args.lambda_input_reconst
        self.lambda_reconst_rereconst = args.lambda_reconst_rereconst
        self.lambda_input_rereconst = args.lambda_input_rereconst

        # z_z_tilda criteria
        self.div_z_z_tilda = args.div_z_z_tilda

    def loss_ae(self, input_x, latent_z, reconst_x, latent_z_tilda, rereconst_x):
        # W(X, X~)
        wasser_input_reconst = self.compute_wasser_ae(input_x, latent_z, reconst_x, self.lambda_mmd)

        # W(X~, X~~)
        wasser_reconst_rereconst = self.compute_wasser_ae(reconst_x.detach(), latent_z_tilda, rereconst_x, self.lambda_mmd)

        # d(X, X~~)
        rerereconst_loss = super().reconstruction_loss(input_x, rereconst_x)

        # d(z, z~) 
        div_z_z_tilda = ASSYMETRIC_DIVERGENCES[self.div_z_z_tilda](latent_z.detach(), latent_z_tilda, self.device)

        loss = -(wasser_input_reconst * self.lambda_input_reconst) + (wasser_reconst_rereconst * self.lambda_reconst_rereconst) + (rerereconst_loss * self.lambda_input_rereconst) + (div_z_z_tilda * self.lambda_z_z_tilda)
        return loss

    def train(self):
        # tensorboard
        if self.tensorboard:
            writer = SummaryWriter()

        train_dataloader = self.env.train

        Tensor = torch.cuda.FloatTensor
        batches_done = 0

        for epoch in range(self.epoches):

            # loop dataloader
            for i, (data, _) in enumerate(train_dataloader):

                # batch data
                input_data_x = autograd.Variable(data.type(Tensor))

                # train info_encoder, decoder
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.second_encoder.zero_grad()
                self.second_decoder.zero_grad()

                # 1st encoder
                latent_z = self.encoder(input_data_x)
                reconst_x = self.decoder(latent_z)

                # 2nd encoder
                latent_z_tilda = self.second_encoder(reconst_x)
                rereconst_x = self.second_decoder(latent_z_tilda)

                ae_loss = self.loss_ae(input_data_x, latent_z, reconst_x, latent_z_tilda, rereconst_x)
                input_rereconst_loss = self.reconstruction_loss(input_data_x, rereconst_x)

                # Optimization
                ae_loss.backward()
                self.optim_E.step()
                self.optim_D.step()
                self.optim_second_E.step()
                self.optim_second_D.step()

                if self.logger is not None:
                    self.logger.info("[Epoch %d/%d] [Batch %d/%d] [AE loss: %f] [input_rereconst loss: %f]"
                            % (epoch, 
                            self.epoches, 
                            i, 
                            len(train_dataloader), 
                            ae_loss.item(), 
                            input_rereconst_loss.item()))
                
                if self.tensorboard:
                    logging_info = {
                        'AE loss': ae_loss.item(),
                        'input_rereconst loss': input_rereconst_loss.item()
                    }

                    # graph
                    for tag, value in logging_info.items():
                        writer.add_scalar(tag, value, batches_done)
                    
                    # images
                    if batches_done % self.sample_interval == 0:
                        tag1 = self.env.env_name + "/" + self.agent + "/" + type(self.encoder).__name__ + "/" + "reconst" + "/%d" %batches_done
                        writer.add_images(tag1, reconst_x.view(64, 1, 28, 28), batches_done)
                        tag2 = self.env.env_name + "/" + self.agent + "/" + type(self.encoder).__name__ + "/" + "rereconst" + "/%d" %batches_done
                        writer.add_images(tag2, rereconst_x.view(64, 1, 28, 28), batches_done)
                
                batches_done += self.n_critic
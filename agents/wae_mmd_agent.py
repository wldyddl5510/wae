from wae_agent import WaeAgent
from utils import calculate_l2_dists, calculate_inner_product, calculate_res
from utils import SAMPLE_PRIOR_DIST, SCALE_LIST, NORM
import torch
import torch.nn as nn

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
    Agent for Wasserstein autoencoder with mmd penelty
"""

class WaeMmdAgent(WaeAgent):


    def __init__(self, args, module, env, device, logger = None):
        super().__init__(args, module, env, device, logger)

        # for mmd loss
        self.lambda_mmd = args.lambda_mmd
        self.c = 2 * self.dim_z * 1.0

        self.criterion = nn.MSELoss()

    def loss_ae(self, input_x, latent_z, reconst_x):
        (ae_loss, _, _) = self.compute_wasser_ae(input_x, latent_z, reconst_x, self.lambda_mmd)
        return ae_loss

    def mmd_penelty(self, latent_z, dim_z, prior, device):
        batch_size = latent_z.size(0)
        z_from_prior = SAMPLE_PRIOR_DIST[self.prior](batch_size, dim_z, device)
        return self.imq_kernel(latent_z, z_from_prior)
        
    def imq_kernel(self, latent_z, prior_z):
        batch_size = self.env.batch_size
        # calculate_elements
        norms_latent_z, l2_dists_latent_z = self.calculate_kernel_elements(latent_z)
        norms_prior_z, l2_dists_prior_z = self.calculate_kernel_elements(prior_z)

        # calculate dot products between actual and prior
        dot_prods_latent_prior = calculate_inner_product(prior_z, latent_z)

        # calculate dist between actual and prior
        l2_dists_latent_prior = calculate_l2_dists(norms_latent_z, norms_prior_z, dot_prods_latent_prior)

        mmd = 0.0
        for scale in SCALE_LIST:
            current_c = self.c * scale
            res1 = calculate_res(current_c, l2_dists_latent_z)
            res2 = calculate_res(current_c, l2_dists_prior_z)
            res3 = calculate_res(current_c, l2_dists_latent_prior)

            #res1 += calculate_res(current_c, dists_prior_z)
            #res1 = (1 - torch.eye(batch_size).to(self.device)) * res1
            #res1 = res1.sum() / (batch_size - 1)
            #res2 = calculate_res(current_c, dists_latent_prior)
            #res2 = res2.sum() * 2. / batch_size
            
            #mmd += res1 - res2
            mmd += torch.mean(res1 + res2 - 2.0 * res3)
        return mmd

    def calculate_kernel_elements(self, z):
        #s^2(z) = sum((z^2)
        norm_z = torch.sum(z ** self.norm, dim = 1, keepdim = True)
        # p2_norm_z = z.pow(2).sum(1).unsqueeze(0)
        # s(z) = sum(z)
        # norm_z = z.sum(1).unsqueeze(0)
        # s(z) * s(z)'
        dotprods_z = calculate_inner_product(z, z)
        # s^2(z) + s^2(z)' - 2s(z)*s(z)'
        l2_dists_z = calculate_l2_dists(norm_z, norm_z, dotprods_z)

        return norm_z, l2_dists_z

    # Compute wasserstein distance between X and X~
    def compute_wasser_ae(self, input_x, latent_z, reconst_x, lambda_mmd):
        mmd_loss = self.mmd_penelty(latent_z, self.dim_z, self.prior, self.device)
        reconst_loss = self.reconstruction_loss(input_x, reconst_x)
        if torch.isnan(mmd_loss) and torch.isnan(reconst_loss):
            print("Both terms are problem!")
            raise RuntimeError
        elif torch.isnan(mmd_loss):
            print("mmd prolem!")
            raise RuntimeError
        elif torch.isnan(reconst_loss):
            print("Reconst problem!")
            raise RuntimeError
        else:
            return reconst_loss - (lambda_mmd * mmd_loss), reconst_loss, mmd_loss
    
    def reconstruction_loss(self, input_x, reconst_x):
        # loss = torch.mean(torch.sum(((reconst_x - input_x.float()) ** self.reconstruct_norm), dim = 1))
        recon_loss = self.criterion(reconst_x, input_x)
        return recon_loss

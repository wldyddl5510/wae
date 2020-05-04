from torch.distributions import Normal, Uniform
import torch
import torch.nn.functional as F
import torch.nn as nn

import pdb

def sample_gaussian(n_sample, dim_sample, device, mu = torch.tensor([0.0]), sigma = torch.tensor([1.0])):
    gaussian_dist = Normal(mu.to(device), sigma.to(device))
    samples = gaussian_dist.sample((n_sample, dim_sample))
    samples = torch.reshape(samples, (n_sample, dim_sample))
    return samples

def sample_uniform(sample_n):
    pass

def calculate_l2_dists(norm_1, norm_2, inner_product):
    return norm_1 + norm_2.t() - 2. * inner_product

def calculate_inner_product(z1, z2):
    return torch.mm(z1, z2.t())

def calculate_res(current_c, dists):
    res = current_c / (current_c + dists)
    return torch.mean(res)

# Calculate divergence of x from y
def kl_div(x, y, device):
    x.to(device); y.to(device)
    return F.kl_div(x, y)

# subspace robust wasserstein
def srw(x, y, dim_z, device):
    pass

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

ASSYMETRIC_DIVERGENCES = {"KL": kl_div}

# Scale list -> for calculating mmd
SCALE_LIST = [.1, .2, .5, 1., 2., 5., 10.]
# Computation settings
EPS = 1e-5
INF = 2e5

NORM = {'L1': 1, 'L2': 2}

# priors
SAMPLE_PRIOR_DIST = {'GAUSSIAN': sample_gaussian, 'UNIFORM': sample_uniform}
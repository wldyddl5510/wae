import os
from torch.utils.data import Dataset, DataLoader

from paths import PROJECT_PATH, DATA_PATH

# For random number generation
import math
import random
import numpy as np

"""
    Dataset class for S^1
    random generation
    (x, y) = (cos t, sin t)
"""
class S1(Dataset):
    

    def __init__(self, root, num_data, train_ratio = 0.9, train = True, download = False):
        theta = np.random.rand(num_data) * (2 * np.pi)
        x, y = np.cos(theta), np.sin(theta)
        dataset = np.array([np.array([x, y]) for x ,y in zip(x, y)])

        num_train = int(num_data * train_ratio)
        self.num_train = num_train
        self.root_dir = root
        self.all_data = dataset
        if train:
            self.dataset = dataset[:num_train]
            self.num_data = num_train
        else:
            self.dataset = dataset[(num_train + 1): ]
            self.num_data = num_data - num_train

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return self.dataset[idx]

    """
        Sample near (0, 0) -> harming global homotopy of the dataset
    """
    def append_anomaly_near_zero(self, ratio_anomaly = 0.05, radius_anomaly = 0.05):
        # TODO: Implement Anomaly data
        shuffled_data = np.random.shuffle(self.all_data)
        num_anomaly = int(len(shuffled_data) * ratio_anomaly)
        normal_data = shuffled_data[num_anomaly: ]

        # Sample from Ball(0, r) for anomaly data
        # sample from the sphere, and then multiply alpha
        theta = np.random.rand(num_anomaly) * (2 * np.pi)
        r = np.random.uniform(0, radius_anomaly, num_anomaly)
        sqrt_r = np.sqrt(r)
        x, y = sqrt_r * np.cos(theta), sqrt_r * np.sin(theta)
        anomaly_dataset = np.array([np.array([x, y]) for x ,y in zip(x, y)])

        self.all_data = np.concatenate(anomaly_dataset, normal_data)
        self.all_data = np.random.permutation(self.all_data)
        self.num_train

def s1(num_data, train_ratio, batch_size, num_workers):

    return {
        'size': 1,
        'train': DataLoader(
            S1(DATA_PATH, num_data, train_ratio = train_ratio, train = True),
            batch_size = batch_size,
            num_workers = num_workers,
            drop_last = True,
            shuffle = True
        ),
        'test': DataLoader(
            S1(DATA_PATH, num_data, train_ratio = train_ratio, train = False),
            batch_size = batch_size,
            num_workers = num_workers,
            drop_last = False, 
            shuffle = False
        )
    }

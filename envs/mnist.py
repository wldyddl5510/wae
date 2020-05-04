import os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from paths import PROJECT_PATH, DATA_PATH

__all__ = ['mnist', 'fashion_mnist']

class MNIST(datasets.MNIST):
    def __init__(self, root, image_shape, train, download_dataset):
        super().__init__(root, train = train, download = download_dataset)
        if train:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(28, 4),
                transforms.Resize(image_shape[1]),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.1307], std = [0.3081])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_shape[1]),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.1307], std = [0.3081])
            ])


class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, root, image_shape, train, download_dataset):
        super().__init__(root, train = train, download = download_dataset)
        if train:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(28, 4),
                transforms.Resize(image_shape[1]),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.1307], std = [0.3081])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_shape[1]),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.1307], std = [0.3081]),
            ])


def mnist(batch_size, num_workers, image_shape, download):
    # Labels = 0 ~ 9
    labels = list(range(10))
    return {
        # 'size': (1, 28, 28),
        'shape': image_shape,
        'labels': labels,
        'batch_size': batch_size,
        'train': DataLoader(
            MNIST(DATA_PATH, image_shape, train = True, download_dataset = download),
            batch_size = batch_size,
            num_workers = num_workers,
            drop_last = True,
            shuffle = True
        ),
        'test': DataLoader(
            MNIST(DATA_PATH, image_shape, train = False, download_dataset = download),
            batch_size = batch_size,
            num_workers = num_workers,
            drop_last = False,
            shuffle = False
        )
    }


def fashion_mnist(batch_size, num_workers, image_shape, download):
    # labels
    labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return {
        'shape': image_shape,
        'labels': labels,
        'batch_size': batch_size,
        'train': DataLoader(
            FashionMNIST(DATA_PATH, image_shape, train = True, download_dataset = download),
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = True,
            drop_last = True,
            shuffle = True
        ),
        'test': DataLoader(
            FashionMNIST(DATA_PATH, image_shape, train = False, download_dataset = download),
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = True,
            drop_last = False,
            shuffle = False
        )
    }
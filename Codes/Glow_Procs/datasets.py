from pathlib import Path

import torch
import torch.nn.functional as F

from torchvision import transforms, datasets
import numpy as np


n_bits = 8


def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2**n_bits
    if n_bits < 8:
      x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x


def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2**n_bits
    return torch.clamp(x, 0, 255).byte()


def get_CIFAR10(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1)),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])

    train_transform = transforms.Compose(transformations)

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), num_classes)

    path = Path(dataroot) / 'data' / 'CIFAR10'
    train_dataset = datasets.CIFAR10(path, train=True,
                                     transform=train_transform,
                                     target_transform=one_hot_encode,
                                     download=download)

    test_dataset = datasets.CIFAR10(path, train=False,
                                    transform=test_transform,
                                    target_transform=one_hot_encode,
                                    download=download)

    return image_shape, num_classes, train_dataset, test_dataset

def get_CIFAR100(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1)),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])

    train_transform = transforms.Compose(transformations)

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), num_classes)

    path = Path(dataroot) / 'data' / 'CIFAR100'
    train_dataset = datasets.CIFAR100(path, train=True,
                                     transform=train_transform,
                                     target_transform=one_hot_encode,
                                     download=download)

    test_dataset = datasets.CIFAR100(path, train=False,
                                    transform=test_transform,
                                    target_transform=one_hot_encode,
                                    download=download)

    return image_shape, num_classes, train_dataset, test_dataset

def get_SVHN(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    transform = transforms.Compose(transformations)

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), num_classes)

    path = Path(dataroot) / 'data' / 'SVHN'
    train_dataset = datasets.SVHN(path, split='train',
                                  transform=transform,
                                  target_transform=one_hot_encode,
                                  download=download)

    test_dataset = datasets.SVHN(path, split='test',
                                 transform=transform,
                                 target_transform=one_hot_encode,
                                 download=download)

    return image_shape, num_classes, train_dataset, test_dataset

def Blankloader(bsize):
    while True:
        r=torch.rand(1)
        im=(torch.ones([bsize,3,32,32])*r)-0.5
        yield (im,torch.tensor(0))

def Whitenoiseloader(bsize):
    while True:
        r=torch.randn([bsize,3,32,32])-0.5
        yield (r,torch.tensor(0))

def Uninoiseloader(bsize):
    while True:
        r=torch.rand([bsize,3,32,32])-0.5
        yield (r,torch.tensor(0))

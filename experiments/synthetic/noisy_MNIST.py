#!/usr/bin/env python3
from torchvision.datasets import MNIST

class Noisy_MNIST:
    def __init__(trainsplit: int):
        transform = transforms.Compose([
            transforms.ToTensor()])
        train = MNIST(".",train=True, download=True, transform=transform)
        test  = MNIST(".",train=False, download=True, transform=transform)

    def get_noisy_trainset():
        pass

    def get_noisy_testset():
        pass

    def get_clean_trainset():
        pass

    def get_clean_testset():
        pass

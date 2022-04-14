#!/usr/bin/env python3
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
import numpy as np
from scipy.stats import norm, chi2

class Noisy_MNIST:
    def __init__(self,trainsplit: int):

        transform = transforms.Compose([
            transforms.ToTensor()])
        train = MNIST(".",train=True, download=True, transform=transform)
        test  = MNIST(".",train=False, download=True, transform=transform)
        self.X = []
        self.y = []

        for i in range(len(train)):
            self.X.append(train[i][0])
            self.y.append(train[i][1])

        for i in range(len(test)):
           self.X.append(test[i][0])
           self.y.append(test[i][1])

        self.X = torch.cat(self.X)
        self.y = torch.Tensor(self.y)

        def sample_chisquared():
            return chi2.rvs(1) * 0.04


        noises = []
        for i in range(10):
            noises.append((norm.rvs(0.2,0.04),sample_chisquared()))

        self.noisy_X = torch.clone(self.X)

        for i in range(self.noisy_X.shape[0]):
            mean, var = noises[int(self.y[i][0])]
            self.noisy_X[i] = self.noisy_X[i] + norm.rvs(mean,var,self.noisy_X[0].shape)


    def get_noisy_trainset():
        pass

    def get_noisy_testset():
        pass

    def get_clean_trainset():
        pass

    def get_clean_testset():
        pass

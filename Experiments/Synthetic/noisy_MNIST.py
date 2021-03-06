#!/usr/bin/env python3
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.stats import norm, chi2
from sklearn.model_selection import train_test_split


class Noisy_MNIST:
    def __init__(self, test_size=0.15):

        try:
            dataset = torch.load("noisy_mnist.dataset")
            self.X = dataset["clean_x"]
            self.noisy_X = dataset["noisy_x"]
            self.y = dataset["clean_y"]
        except:
            transform = transforms.Compose([transforms.ToTensor()])
            train = MNIST(".", train=True, download=True, transform=transform)
            test = MNIST(".", train=False, download=True, transform=transform)
            self.X = []
            self.y = []

            for i in range(len(train)):
                self.X.append(train[i][0])
                self.y.append(train[i][1])

            for i in range(len(test)):
                self.X.append(test[i][0])
                self.y.append(test[i][1])

            self.X = torch.cat(self.X)
            self.y = torch.Tensor(self.y).type(torch.LongTensor)

            def sample_chisquared():
                return chi2.rvs(1) * 0.04

            noises = []
            for i in range(10):
                noises.append((norm.rvs(0.2, 0.04), sample_chisquared()))

            self.X = torch.reshape(
                self.X, (self.X.shape[0], 1, self.X.shape[1], self.X.shape[2])
            )
            self.noisy_X = torch.clone(self.X)

            for i in range(self.noisy_X.shape[0]):
                mean, var = noises[int(self.y[i].item())]
                self.noisy_X[i] = self.noisy_X[i] + norm.rvs(
                    mean, var, self.noisy_X[0].shape
                )

            dataset_dict = {
                "noisy_x": self.noisy_X,
                "clean_x": self.X,
                "clean_y": self.y,
            }
            torch.save(dataset_dict, "noisy_mnist.dataset")

        (
            self.noisy_X_train,
            self.noisy_X_test,
            self.noisy_y_train,
            self.noisy_y_test,
        ) = train_test_split(self.noisy_X, self.y, test_size=test_size, random_state=42)
        (
            self.clean_X_train,
            self.clean_X_test,
            self.clean_y_train,
            self.clean_y_test,
        ) = train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    def get_noisy_trainset(self):
        return self.noisy_X_train, self.noisy_y_train

    def get_noisy_testset(self):
        return self.noisy_X_test, self.noisy_y_test

    def get_clean_trainset(self):
        return self.clean_X_train, self.clean_y_train

    def get_clean_testset(self):
        return self.clean_X_test, self.clean_y_test


class MNISTDataset(Dataset):
    def __init__(self, mnist_data, train=True, noisy=True):
        self.data = None
        if train and noisy:
            self.data = mnist_data.get_noisy_trainset()
        elif train and not noisy:
            self.data = mnist_data.get_clean_trainset()
        elif not train and noisy:
            self.data = mnist_data.get_noisy_testset()
        elif not train and not noisy:
            self.data = mnist_data.get_clean_testset()

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

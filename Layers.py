import torch.nn as nn
import torch.nn.functional as F
from klein import generate_klein_filter, generate_pc_filter
import numpy as np
import torch


class KF_Layer(nn.Module):
    def __init__(self, size: int, slices: int):
        # Need to create sqrt(slices) evenly spaced angles for both thetas
        angle = int(np.sqrt(slices))
        thetas1 = [(i * np.pi) / angle for i in range(angle)]
        thetas2 = [(i * np.pi * 2) / angle for i in range(angle)]
        self.filters = []

        for theta2 in thetas2:
            for theta1 in thetas1:
                self.filters.append(generate_klein_filter(size, theta1, theta2))

    def forward(self, x):
        output = []
        for filter in self.filters:
            nb_channels = 1
            filter = torch.tensor(filter, dtype=torch.float32)
            filter = filter.view(1, 1, filter.shape[0], filter.shape[1]).repeat(
                1, nb_channels, 1, 1
            )
            output.append(F.conv2d(x, filter).squeeze(0))
        return torch.stack(output)


class CF_Layer(nn.Module):
    def __init__(self, size: int, slices: int):
        # Need to generate slices number of evenly spaced angles around a circle
        thetas = [(i * np.pi * 2) / slices for i in range(slices)]

        self.filters = []

        for theta in thetas:
            self.filters.append(generate_pc_filter(size, theta))

    def forward(self, x):
        output = []
        for filter in self.filters:
            nb_channels = 1
            filter = torch.tensor(filter, dtype=torch.float32)
            filter = filter.view(1, 1, filter.shape[0], filter.shape[1]).repeat(
                1, nb_channels, 1, 1
            )
            output.append(F.conv2d(x, filter).squeeze(0))
        return torch.stack(output)

import torch
import torch.nn as nn
import torch.nn.functional as F


class NOL_NOL(nn.Module):
    def __init__(self, conv_slices: int, kernel_size: int, num_classes: int, image_dim: tuple):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, conv_slices, kernel_size),
            nn.ReLU(),
            nn.Conv2d(conv_slices, conv_slices, kernel_size),
            nn.ReLU())

        out = self.convs(torch.randn(image_dim).view(-1, 1, image_dim[0], image_dim[1]))
        self.input_size = out[0].shape[0] * out[0].shape[1] * out[0].shape[2]

        self.fcs = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.input_size)
        x = self.fcs(x)
        return F.softmax(x, dim=1)
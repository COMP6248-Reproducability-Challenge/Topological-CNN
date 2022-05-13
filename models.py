import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import KF_Layer, CF_Layer

class NOL(nn.Module):
    def __init__(
        self, conv_slices: int, kernel_size: int, num_classes: int, image_dim: tuple
    ):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, conv_slices, kernel_size),
            nn.ReLU(),
        )

        out = self.convs(torch.randn(image_dim).view(-1, 1, image_dim[0], image_dim[1]))
        self.input_size = out[0].shape[0] * out[0].shape[1] * out[0].shape[2]

        self.fcs = nn.Sequential(
            nn.Linear(self.input_size, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.input_size)
        x = self.fcs(x)
        if (
            not self.training
        ):
            x = F.softmax(x, dim=1)
        return x


class NOL_NOL(nn.Module):
    def __init__(
        self, conv_slices: int, kernel_size: int, num_classes: int, image_dim: tuple
    ):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, conv_slices, kernel_size),
            nn.ReLU(),
            nn.Conv2d(conv_slices, conv_slices, kernel_size),
            nn.ReLU(),
        )

        out = self.convs(torch.randn(image_dim).view(-1, 1, image_dim[0], image_dim[1]))
        self.input_size = out[0].shape[0] * out[0].shape[1] * out[0].shape[2]

        self.fcs = nn.Sequential(
            nn.Linear(self.input_size, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.input_size)
        x = self.fcs(x)
        if (
            not self.training
        ):  # nn.CrossEntropyLoss implicitly adds a softmax before a logarithmic loss
            x = F.softmax(x, dim=1)
        return x


class NOL_NOL_POOL(nn.Module):
    def __init__(
        self, conv_slices: int, kernel_size: int, num_classes: int, image_dim: tuple
    ):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, conv_slices, kernel_size),
            nn.ReLU(),
            nn.Conv2d(conv_slices, conv_slices, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        out = self.convs(torch.randn(image_dim).view(-1, 1, image_dim[0], image_dim[1]))
        self.input_size = out[0].shape[0] * out[0].shape[1] * out[0].shape[2]

        self.fcs = nn.Sequential(
            nn.Linear(self.input_size, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.input_size)
        x = self.fcs(x)
        if (
            not self.training
        ):  # nn.CrossEntropyLoss implicitly adds a softmax before a logarithmic loss
            x = F.softmax(x, dim=1)
        return x


class KF_KF(nn.Module):
    def __init__(
        self, conv_slices: int, kernel_size: int, num_classes: int, image_dim: tuple
    ):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, conv_slices, kernel_size),
            nn.ReLU(),
            nn.Conv2d(conv_slices, conv_slices, kernel_size),
            nn.ReLU(),
        )

        out = self.convs(torch.randn(image_dim).view(-1, 1, image_dim[0], image_dim[1]))
        self.input_size = out[0].shape[0] * out[0].shape[1] * out[0].shape[2]

        kf_filters = KF_Layer(kernel_size, conv_slices).filters

        with torch.no_grad():
            for i, filter in enumerate(kf_filters):
                self.convs[0].weight[i] = torch.nn.Parameter(torch.tensor(filter))
                self.convs[2].weight[i] = torch.nn.Parameter(torch.tensor(filter))

        self.fcs = nn.Sequential(
            nn.Linear(self.input_size, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

        # Freeze KF layers
        self.convs[0].requires_grad = False
        self.convs[2].requires_grad = False

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.input_size)
        x = self.fcs(x)
        if (
            not self.training
        ):
            x = F.softmax(x, dim=1)
        return x


class KF_KF_POOL(nn.Module):
    def __init__(
        self, conv_slices: int, kernel_size: int, num_classes: int, image_dim: tuple
    ):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, conv_slices, kernel_size),
            nn.ReLU(),
            nn.Conv2d(conv_slices, conv_slices, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        out = self.convs(torch.randn(image_dim).view(-1, 1, image_dim[0], image_dim[1]))
        self.input_size = out[0].shape[0] * out[0].shape[1] * out[0].shape[2]

        kf_filters = KF_Layer(kernel_size, conv_slices).filters

        with torch.no_grad():
            for i, filter in enumerate(kf_filters):
                self.convs[0].weight[i] = torch.nn.Parameter(torch.tensor(filter))
                self.convs[2].weight[i] = torch.nn.Parameter(torch.tensor(filter))

        self.fcs = nn.Sequential(
            nn.Linear(self.input_size, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

        self.convs[0].requires_grad = False
        self.convs[2].requires_grad = False

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.input_size)
        x = self.fcs(x)
        if (
            not self.training
        ):
            x = F.softmax(x, dim=1)
        return x


class KF_NOL(nn.Module):
    def __init__(
        self, conv_slices: int, kernel_size: int, num_classes: int, image_dim: tuple
    ):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, conv_slices, kernel_size),
            nn.ReLU(),
            nn.Conv2d(conv_slices, conv_slices, kernel_size),
            nn.ReLU(),
        )

        out = self.convs(torch.randn(image_dim).view(-1, 1, image_dim[0], image_dim[1]))
        self.input_size = out[0].shape[0] * out[0].shape[1] * out[0].shape[2]

        # Replace first layer's weights to KF filters
        kf_filters = KF_Layer(kernel_size, conv_slices).filters

        with torch.no_grad():
            for i, filter in enumerate(kf_filters):
                self.convs[0].weight[i] = torch.nn.Parameter(torch.tensor(filter))

        self.fcs = nn.Sequential(
            nn.Linear(self.input_size, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

        self.convs[0].requires_grad = False

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.input_size)
        x = self.fcs(x)
        if (
            not self.training
        ):
            x = F.softmax(x, dim=1)
        return x


class CF_NOL(nn.Module):
    def __init__(
        self, conv_slices: int, kernel_size: int, num_classes: int, image_dim: tuple
    ):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, conv_slices, kernel_size),
            nn.ReLU(),
            nn.Conv2d(conv_slices, conv_slices, kernel_size),
            nn.ReLU(),
        )

        out = self.convs(torch.randn(image_dim).view(-1, 1, image_dim[0], image_dim[1]))
        self.input_size = out[0].shape[0] * out[0].shape[1] * out[0].shape[2]

        cf_filters = CF_Layer(kernel_size, conv_slices).filters

        with torch.no_grad():
            for i, filter in enumerate(cf_filters):
                self.convs[0].weight[i] = torch.nn.Parameter(torch.tensor(filter))

        self.fcs = nn.Sequential(
            nn.Linear(self.input_size, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.input_size)
        x = self.fcs(x)
        if (
            not self.training
        ):
            x = F.softmax(x, dim=1)
        return x

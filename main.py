#!/usr/bin/env python
import klein
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from Layers import KF_Layer, CF_Layer
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # There are probably some combinations of thetas that give identical filters
    # so we can reduce this list
    #thetas = [0, np.pi, -np.pi, np.pi/2, -np.pi/2, np.pi/4, -np.pi/4]
    thetas = [(i*np.pi)/8 for i in range(16)]
    thetas = [(i*np.pi)/2 for i in range(4)]
    #klein.display_kernels(thetas, 5, circle=False)

    # This is just to check if the forward pass of the layers work
    # using one sample of training image from the MINST; this is quite slow!
    transform = transforms.Compose([
        transforms.ToTensor()  # convert to tensor
    ])
    trainset = MNIST(".", train=True, download=True, transform=transform)
    size, slices = 5, 16
    Layer = KF_Layer(size, slices)
    output = Layer.forward(trainset[0][0])
    fig, axs = plt.subplots(slices, 1, figsize=(slices * 2, slices * 2))
    for i in range(slices):
        axs[i].imshow(output[i], cmap=plt.get_cmap('gray'))
    plt.show()



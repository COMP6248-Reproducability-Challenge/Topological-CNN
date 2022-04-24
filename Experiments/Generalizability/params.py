import json
import os
import time

params = {
    "experiment_type": "MINST+SVHN",  # 'SVHN+MINST', 'Kaggle+CIFAR-10', 'CIFAR-10+Kaggle'
    "model": "NOL+NOL",  # 'NOL+NOL+pooled', 'KF+KF', 'KF+KF+pooled'
    "checkpoint": time.strftime("%Y_%m_%d_%H_%M_%S"),
    "epochs": 5,
    "batch_size": 100,
    "image_dim": (28, 28),
    "learning_rate": 1e-5,
    "conv_slices": 64,
    "kernel_size": 3,
    "num_classes": 10,
    "device": "cpu",  # 'cuda' or 'cpu'
}

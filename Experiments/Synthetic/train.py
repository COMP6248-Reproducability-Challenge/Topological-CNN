#!/usr/bin/env python3
import sys

sys.path.append("../../")
import torchbearer
from models import NOL_NOL, KF_NOL, CF_NOL
import torch.optim as optim
from noisy_MNIST import Noisy_MNIST, MNISTDataset
import torch.nn as nn
from torch.utils.data import Dataset
import torch
from torch.utils.data.dataloader import DataLoader
import json
import time

EPOCHS = 5
LR = 1e-5
BATCH_SIZE = 100
dataset = Noisy_MNIST(0.15)
KERNEL_SIZE = 3
CONV_SLICES = 64
IMAGE_SIZE = (28, 28)
NUM_CLASSES = 10


def train(model, train_loader, test_loader, device, epochs=1):
    model.train()
    log_interval = 10

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LR)

    results = {"training_loss": [], "testing_accuracy": [], "training_time": 0, "testing_time":0}

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            train_start = time.perf_counter()
            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, target)
            loss.backward()
            optimiser.step()
            train_stop = time.perf_counter()

            results["training_time"] += train_start - train_stop

            if batch_idx % log_interval == 0:
                results["training_loss"].append(loss.item())
                test_start = time.perf_counter()
                accuracy = test(model, test_loader)
                test_stop = time.perf_counter()
                results["testing_time"] += test_start - test_stop
                results["testing_accuracy"].append(accuracy)

                print(f"Epoch {epoch}: batch {batch_idx}, test accuracy {accuracy}")

                model.train()
    return results


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            correct += (outputs.argmax(dim=1) == target).type(torch.float).sum().item()
            total += data.shape[0]

    return (100.0 * correct) / total


def save_model(model, model_name):
    torch.save(
        model.state_dict(),
        f'Checkpoint/{time.strftime("%Y_%m_%d_%H_%M_%S")}_Synthetic_{model_name}_weights',
    )


if __name__ == "__main__":

    print("Generating Dataset ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = Noisy_MNIST()

    noisy_trainset = MNISTDataset(raw_data)
    noisy_testset = MNISTDataset(raw_data, train=False)

    clean_trainset = MNISTDataset(raw_data, noisy=False)
    clean_testset = MNISTDataset(raw_data, noisy=False, train=False)

    print("Creating Dataloaders ...")
    noisy_trainloader = DataLoader(noisy_trainset, batch_size=BATCH_SIZE, shuffle=True)
    noisy_testloader = DataLoader(noisy_testset, batch_size=BATCH_SIZE, shuffle=True)

    clean_trainloader = DataLoader(clean_trainset, batch_size=BATCH_SIZE, shuffle=True)
    clean_testloader = DataLoader(clean_testset, batch_size=BATCH_SIZE, shuffle=True)

    models = [
        (
            "NOL+NOL",
            NOL_NOL(CONV_SLICES, KERNEL_SIZE, NUM_CLASSES, IMAGE_SIZE).to(device),
            NOL_NOL(CONV_SLICES, KERNEL_SIZE, NUM_CLASSES, IMAGE_SIZE).to(device),
        ),
        (
            "KF+NOL",
            KF_NOL(CONV_SLICES, KERNEL_SIZE, NUM_CLASSES, IMAGE_SIZE).to(device),
            KF_NOL(CONV_SLICES, KERNEL_SIZE, NUM_CLASSES, IMAGE_SIZE).to(device),
        ),
        (
            "CF+NOL",
            CF_NOL(CONV_SLICES, KERNEL_SIZE, NUM_CLASSES, IMAGE_SIZE).to(device),
            CF_NOL(CONV_SLICES, KERNEL_SIZE, NUM_CLASSES, IMAGE_SIZE).to(device),
        ),
    ]
    # , ('CF+NOL', CF_NOL(CONV_SLICES,KERNEL_SIZE,NUM_CLASSES,IMAGE_SIZE).to(device))
    # models = [('NOL+NOL',NOL_NOL(CONV_SLICES,KERNEL_SIZE,NUM_CLASSES,IMAGE_SIZE).to(device))]
    experiment_results = {}

    print("Training ...")
    for model_id, clean_model, noisy_model in models:
        print("Training: " + model_id)
        experiment_results[model_id + "_noisy_test"] = train(
            clean_model, clean_trainloader, noisy_testloader, device, EPOCHS
        )
        save_model(clean_model, model_id + "_noisy_test")
        experiment_results[model_id + "_noisy_train"] = train(
            noisy_model, noisy_trainloader, clean_testloader, device, EPOCHS
        )
        save_model(clean_model, model_id + "_noisy_train")

    with open(
        f'Checkpoint/{time.strftime("%Y_%m_%d_%H_%M_%S")}_Synthetic_results.json',
        "w",
    ) as f:
        json.dump(experiment_results, f)

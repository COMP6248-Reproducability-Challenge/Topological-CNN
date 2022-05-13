#!/usr/bin/env python3
import sys

sys.path.append("../../")
import torchbearer
from models import NOL
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets
import json
import time

EPOCHS = 1
LR = 1e-4
BATCH_SIZE = 100
KERNEL_SIZE = 5
CONV_SLICES = 16
IMAGE_SIZE = (28, 28)
NUM_CLASSES = 10


def train(model, train_loader, test_loader, device, epochs=1):
    model.train()
    log_interval = 10

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LR)

    results = {"training_loss": [], "testing_accuracy": [], "training_time": 0, "testing_time": 0}

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
        f'Checkpoint/{time.strftime("%Y_%m_%d_%H_%M_%S")}_Interpretability_{model_name}_weights',
    )


if __name__ == "__main__":
    print("Generating Dataset ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = datasets.MNIST(
        ".",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    testset = datasets.MNIST(
        ".",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    merge_data = torch.utils.data.ConcatDataset([trainset, testset])

    new_train_set, new_test_set = torch.utils.data.random_split(torch.utils.data.ConcatDataset([trainset, testset]),
                                                                [int(len(merge_data) * 0.85),
                                                                 int(len(merge_data) * 0.15)],
                                                                generator=torch.Generator().manual_seed(0))

    print("Creating Dataloaders ...")
    trainloader = DataLoader(new_train_set, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(new_test_set, batch_size=BATCH_SIZE, shuffle=True)

    model_id = "NOL"
    model = NOL(CONV_SLICES, KERNEL_SIZE, NUM_CLASSES, IMAGE_SIZE).to(device)

    experiment_results = {}

    print("Training: " + model_id)
    experiment_results[model_id + "MINST"] = train(
        model, trainloader, testloader, device, EPOCHS
    )
    save_model(model, model_id + "MINST")

    with open(
            f'Checkpoint/{time.strftime("%Y_%m_%d_%H_%M_%S")}_Interpretability_results.json',
            "w",
    ) as f:
        json.dump(experiment_results, f)

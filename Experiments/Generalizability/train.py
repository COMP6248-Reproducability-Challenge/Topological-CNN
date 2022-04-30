import sys

sys.path.append("../../")
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import Experiments.Generalizability.data as data
from models import KF_KF, KF_KF_POOL, NOL_NOL, NOL_NOL_POOL


EPOCHS = 5
LR = 1e-5
BATCH_SIZE = 100
KERNEL_SIZE = 3
CONV_SLICES = 64

def train(model, train_loader, test_loader, device, epochs=1):
    model.train()
    log_interval = 10

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LR)

    results = {"training_loss": [], "testing_accuracy": []}

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, target)
            loss.backward()
            optimiser.step()

            if batch_idx % log_interval == 0:
                results["training_loss"].append(loss.item())
                accuracy = test(model, test_loader)
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
        f'Checkpoint/{time.strftime("%Y_%m_%d_%H_%M_%S")}_Generalisation_{model_name}_weights',
    )

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment on MINST and SVHN
    MINST_dataset = data.load_MINST(False)
    MINST_loader = DataLoader(
        MINST_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    SVHN_dataset = data.load_SVHN(False)
    SVHN_loader = DataLoader(
        SVHN_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    models = [
        (
            "KF+KF",
            KF_KF(CONV_SLICES, KERNEL_SIZE, 10, (28, 28)).to(device),
            KF_KF(CONV_SLICES, KERNEL_SIZE, 10, (28, 28)).to(device)
        ),
        (
            "KF+KF_pooled",
            KF_KF_POOL(CONV_SLICES, KERNEL_SIZE, 10, (28, 28)).to(device),
            KF_KF_POOL(CONV_SLICES, KERNEL_SIZE, 10, (28, 28)).to(device)
        ),
        (
            "NOL+NOL",
            NOL_NOL(CONV_SLICES, KERNEL_SIZE, 10, (28, 28)).to(device),
            NOL_NOL(CONV_SLICES, KERNEL_SIZE, 10, (28, 28)).to(device)
        ),
        (
            "NOL+NOL_pooled",
            NOL_NOL_POOL(CONV_SLICES, KERNEL_SIZE, 10, (28, 28)).to(device),
            NOL_NOL_POOL(CONV_SLICES, KERNEL_SIZE, 10, (28, 28)).to(device)
        )
    ]

    experiment_results = {}
    print("Training...")
    for model_id, minst_model, svhn_model in models:
        print("Training: " + model_id)
        experiment_results[model_id + "_MINST_train_SVHN_test"] = train(
            minst_model, MINST_loader, SVHN_loader, device, EPOCHS
        )
        save_model(minst_model, model_id + "_MINST_train_SVHN_test")
        experiment_results[model_id + "_SVHN_train_MINST_test"] = train(
            svhn_model, SVHN_loader, MINST_loader, device, EPOCHS
        )
        save_model(svhn_model, model_id + "_SVHN_train_MINST_test")

    with open(
            f'Checkpoint/{time.strftime("%Y_%m_%d_%H_%M_%S")}_Generalisation_results.json',
            "w",
    ) as f:
        json.dump(experiment_results, f)

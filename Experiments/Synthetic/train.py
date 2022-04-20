#!/usr/bin/env python3
import torchbearer
from models import NOL_NOL, KF_NOL, CF_NOL
import torch.optim as optim
from noisy_MNIST import Noisy_MNIST, MNISTDataset
import torch.nn as nn
from torch.utils.data import Dataset

EPOCHS = 5
LR = 1e-5
BATCH_SIZE = 100
dataset = Noisy_MNIST(0.15)
KERNEL_SIZE = 3
CONV_SLICES = 64
IMAGE_SIZE = (28,28)
NUM_CLASSES = 10

def train(model, train_loader, test_loader, device, epochs=1):
    model.train()
    log_interval = 10

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LR)
    
    results = {
	'training_loss': [],
	'testing_accuracy': []
    }

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, target)
            loss.backward()
            optimiser.step()

            if batch_idx % log_interval == 0:
                results['training_loss'].append(loss.item())
                accuracy = test(model, test_loader)
                results['testing_accuracy'].append(accuracy)

                print(f"Epoch {epoch}: batch {batch_idx}, generalisation accuracy {accuracy}")

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

    return ((100.0 * correct) / total)

def save_model(model):
    torch.save(model.state_dict(), f"Checkpoint/{params['checkpoint']}_{params['experiment_type']}_{params['model']}_weights")

    with open(f'Checkpoint/{params["checkpoint"]}_{params["experiment_type"]}_{params["model"]}_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = Noisy_MNIST()

    noisy_trainset = MNISTDataset(raw_data)
    noisy_testset = MNISTDataset(raw_data, train=False)

    clean_trainset = MNISTDataset(raw_data, noisy=False)
    clean_testset = MNISTDataset (raw_data, noisy=False, train=False)

    noisy_trainloader = Dataloader(noisy_trainset, batch_size = BATCH_SIZE, shuffle=True)
    noisy_testloader =  Dataloader(noisy_testset, batch_size = BATCH_SIZE, shuffle=True)

    clean_trainloader = Dataloader(clean_trainset, batch_size = BATCH_SIZE, shuffle=True)
    clean_testloader =  Dataloader(clean_testset, batch_size = BATCH_SIZE, shuffle=True)

    models = [('NOL+NOL',NOL_NOL(CONV_SLICES,KERNEL_SIZE,NUM_CLASSES,IMAGE_SIZE).to(device)), ('KF+NOL', KF_NOL(CONV_SLICES,KERNEL_SIZE,NUM_CLASSES,IMAGE_SIZE).to(device)), ('CF+NOL', CF_NOL(CONV_SLICES,KERNEL_SIZE,NUM_CLASSES,IMAGE_SIZE).to(device))]
    experiment_results = {}

    for model_id, model in models:
        experiment_results[model_id] = train(model, clean_trainloader, noisy_testloader, device, EPOCHS )

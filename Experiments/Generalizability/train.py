import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import Experiments.Generalizability.data as data
from models import NOL_NOL
from params import params

results = {
    'parameters': params,
    'training_loss': [],
    'generalisation_accuracy': [],
    'train_time': 0
}

def train(model, train_loader, test_loader, optimiser, loss_function, device, epochs=1):
    model.train()
    log_interval = 10
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
                results['generalisation_accuracy'].append(accuracy)

                print(f"Epoch {epoch}: batch {batch_idx}, generalisation accuracy {accuracy}")

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

    print('Test Accuracy: %2.2f %%' % ((100.0 * correct) / total))

    return ((100.0 * correct) / total)

def save_model(model):
    torch.save(model.state_dict(), f"Checkpoint/{params['checkpoint']}_{params['experiment_type']}_{params['model']}_weights")

    with open(f'Checkpoint/{params["checkpoint"]}_{params["experiment_type"]}_{params["model"]}_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":

    # let's try training on MINST using NOL+NOL model first
    # TODO: create a configuration file for different parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_dataset = data.load_MINST(False)
    train_loader = DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=True)

    test_dataset = data.load_SVHN(False)
    test_loader = DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=True)

    model = NOL_NOL(params['conv_slices'], params['kernel_size'], params['num_classes'], params['image_dim']).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_start_time = time.time()
    #train(model, train_loader, test_loader, optimiser, loss_function, device, epochs=5)
    train_time = datetime.timedelta(0, time.time() - train_start_time)
    results['train_time'] = str(train_time)
    print(f'Training completed in {train_time}')
    # save_model(model)

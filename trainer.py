import torch
import numpy as np

@torch.compile()
def trainOneEpoch(model, loader, criterion, optimizer, device):
    model.train()
    runningLoss = 0.0
    correct = 0
    total = 0

    for batchIdx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Stats
        runningLoss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epochLoss = runningLoss / total
    epochAcc = 100. * correct / total
    return epochLoss, epochAcc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    runningLoss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        runningLoss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avgLoss = runningLoss / total
    acc = 100. * correct / total
    return avgLoss, acc

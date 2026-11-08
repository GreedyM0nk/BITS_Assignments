import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .reproducibility import seed_everything


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def train_epoch(model, loader, criterion, optimizer, device, clip=None):
    model.train()
    total, seen = 0.0, 0
    start = time.perf_counter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total += loss.item() * x.size(0)
        seen += x.size(0)
    return total / seen, time.perf_counter() - start


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    truth, predictions = [], []
    for x, y in loader:
        predictions.extend(model(x.to(device)).argmax(-1).cpu().numpy())
        truth.extend(y.numpy())
    return {
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, average="macro"),
        "confusion": confusion_matrix(truth, predictions),
    }


def train_classifier(model, train_loader, valid_loader, test_loader, epochs, lr, device, seed, clip=None):
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    times, history = [], defaultdict(list)
    for _ in range(epochs):
        loss, seconds = train_epoch(model, train_loader, criterion, optimizer, device, clip)
        times.append(seconds)
        history["train_loss"].append(loss)
        history["valid_accuracy"].append(evaluate_classifier(model, valid_loader, device)["accuracy"])
    result = evaluate_classifier(model, test_loader, device)
    result.update(history=history, sec_per_epoch=float(np.mean(times)), params=count_parameters(model))
    return result


@torch.no_grad()
def evaluate_regressor(model, loader, device):
    model.eval()
    losses = []
    for x, y in loader:
        losses.append(torch.nn.functional.mse_loss(model(x.to(device)), y.to(device)).item() * x.size(0))
    return sum(losses) / len(loader.dataset)


def train_regressor(model, train_loader, valid_loader, test_loader, epochs, lr, device, seed, clip=None):
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    history, times = [], []
    for _ in range(epochs):
        loss, seconds = train_epoch(model, train_loader, criterion, optimizer, device, clip)
        history.append((loss, evaluate_regressor(model, valid_loader, device)))
        times.append(seconds)
    return {
        "test_mse": evaluate_regressor(model, test_loader, device),
        "history": history,
        "sec_per_epoch": float(np.mean(times)),
        "params": count_parameters(model),
    }

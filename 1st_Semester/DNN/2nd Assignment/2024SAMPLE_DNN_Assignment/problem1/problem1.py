"""Problem 1 runner.

The notebook imports this module so the same code can be tested from a clean
Python process. Set FAST_MODE to False for the full rubric experiment matrix.
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import FIGURES_1, RESULTS, SETTINGS
from src.data import loaders_from_arrays, load_har, make_adding, validate_adding
from src.models import RecurrentClassifier, RecurrentRegressor
from src.reproducibility import device, seed_everything
from src.results import append_result
from src.training import train_classifier, train_regressor


FAST_MODE = True


def build_adding_loaders(T, batch_size=SETTINGS.batch_size, fast=FAST_MODE):
    n_train, n_valid, n_test = (512, 128, 128) if fast else (10000, 1000, 1000)
    train = make_adding(n_train, T, 0)
    valid = make_adding(n_valid, T, 1)
    test = make_adding(n_test, T, 2)
    validate_adding(*train)
    return tuple(loaders_from_arrays(x, y, batch_size) for x, y in (train, valid, test))


def har_loaders(seed):
    arrays = load_har(ROOT / "data", seed)
    x_train, y_train, x_valid, y_valid, x_test, y_test = arrays[:6]
    return (loaders_from_arrays(x_train, y_train),
            loaders_from_arrays(x_valid, y_valid, shuffle=False),
            loaders_from_arrays(x_test, y_test, shuffle=False))


def plot_class_balance(y, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3))
    values, counts = np.unique(y, return_counts=True)
    plt.bar(values + 1, counts)
    plt.xlabel("Activity class")
    plt.ylabel("Training windows")
    plt.title("Figure 1. Subject-wise HAR training class balance")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_har(fast=FAST_MODE):
    epochs = 2 if fast else SETTINGS.epochs
    hidden = SETTINGS.hidden_size
    all_results = []
    for seed in SETTINGS.seeds:
        train_loader, valid_loader, test_loader = har_loaders(seed)
        for kind in ("RNN", "LSTM", "GRU"):
            result = train_classifier(
                RecurrentClassifier(9, hidden, 6, kind),
                train_loader, valid_loader, test_loader, epochs,
                SETTINGS.lr, device(), seed,
            )
            append_result(RESULTS, {
                "problem": "P1-HAR", "model": kind, "seed": seed,
                "test_acc": result["accuracy"], "macro_f1": result["macro_f1"],
                "params": result["params"], "sec_per_epoch": result["sec_per_epoch"],
                "notes": f"hidden={hidden};epochs={epochs}",
            })
            all_results.append((kind, seed, result))
    best = max(all_results, key=lambda item: item[2]["accuracy"])
    ConfusionMatrixDisplay(best[2]["confusion"]).plot()
    FIGURES_1.mkdir(parents=True, exist_ok=True)
    plt.title(f"Figure 2. Best HAR confusion matrix ({best[0]}, seed {best[1]})")
    plt.tight_layout()
    plt.savefig(FIGURES_1 / "p1_har_confusion_matrix.png", dpi=160)
    plt.close()
    return all_results


def run_adding(fast=FAST_MODE):
    lengths = (50, 100) if fast else (50, 100, 200, 400)
    epochs = 2 if fast else SETTINGS.epochs
    rows = []
    for T in lengths:
        loaders = build_adding_loaders(T, fast=fast)
        for seed in SETTINGS.seeds:
            for kind in ("RNN", "LSTM", "GRU"):
                result = train_regressor(
                    RecurrentRegressor(2, SETTINGS.hidden_size, kind),
                    *loaders, epochs, SETTINGS.lr, device(), seed,
                )
                append_result(RESULTS, {
                    "problem": "P1-adding", "model": kind, "seed": seed,
                    "seq_len": T, "test_mse": result["test_mse"],
                    "params": result["params"], "sec_per_epoch": result["sec_per_epoch"],
                    "notes": f"baseline=0.167;solved={result['test_mse'] < 0.05}",
                })
                rows.append((T, kind, seed, result))
    return rows


def run_smoke():
    seed_everything(SETTINGS.seeds[0])
    x, y = make_adding(32, 50, 0)
    validate_adding(x, y)
    return {"device": str(device()), "adding_validation": "passed"}


if __name__ == "__main__":
    print(run_smoke())
    if "--full" in sys.argv:
        run_har(False)
        run_adding(False)

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
        if seed == SETTINGS.seeds[0]:
            arrays = load_har(ROOT / "data", seed)
            plot_class_balance(arrays[1], FIGURES_1 / "p1_class_balance.png")
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


def gradient_profile(model, x, y, device_name, before_training=True):
    model = model.to(device_name)
    model.eval()
    x, y = x.to(device_name), y.to(device_name)
    x.requires_grad_(True)
    prediction = model(x)
    loss = torch.nn.functional.mse_loss(prediction, y)
    loss.backward()
    norms = x.grad.detach().norm(dim=-1).mean(0).cpu().numpy()
    return norms


def run_gradient_study(fast=FAST_MODE):
    n = 64 if fast else 256
    x, y = make_adding(n, 100, SETTINGS.seeds[0])
    profiles = {}
    for kind in ("RNN", "LSTM", "GRU"):
        model = RecurrentRegressor(2, SETTINGS.hidden_size, kind)
        before = gradient_profile(model, torch.from_numpy(x), torch.from_numpy(y), device())
        loaders = tuple(loaders_from_arrays(*pair) for pair in (
            make_adding(512 if fast else 10000, 100, 0),
            make_adding(128 if fast else 1000, 100, 1),
            make_adding(128 if fast else 1000, 100, 2),
        ))
        trained = train_regressor(model, *loaders, 1 if fast else SETTINGS.epochs,
                                  SETTINGS.lr, device(), SETTINGS.seeds[0])
        after = gradient_profile(model, torch.from_numpy(x), torch.from_numpy(y), device())
        profiles[kind] = (before, after)
    FIGURES_1.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for kind, (before, after) in profiles.items():
        plt.plot(np.arange(100), before, "--", label=f"{kind} before")
        plt.plot(np.arange(100), after, label=f"{kind} after")
    plt.yscale("log")
    plt.xlabel("Time step")
    plt.ylabel(r"$||\partial L / \partial x_t||$")
    plt.title("Figure 3. Gradient norm before and after training")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_1 / "p1_gradient_norms.png", dpi=160)
    plt.close()
    return profiles


def run_hidden_size_sweep(fast=FAST_MODE):
    sizes = (16, 32) if fast else (16, 32, 64, 128)
    train, valid, test = har_loaders(SETTINGS.seeds[0])
    rows = []
    for size in sizes:
        for kind in ("LSTM", "GRU"):
            result = train_classifier(
                RecurrentClassifier(9, size, 6, kind), train, valid, test,
                1 if fast else SETTINGS.epochs, SETTINGS.lr, device(), SETTINGS.seeds[0],
            )
            rows.append((size, kind, result["accuracy"], result["params"]))
    frame = np.array([(size, acc) for size, _, acc, _ in rows])
    plt.figure(figsize=(6, 4))
    for kind in ("LSTM", "GRU"):
        subset = [(s, a) for s, k, a, _ in rows if k == kind]
        plt.plot([x[0] for x in subset], [x[1] for x in subset], marker="o", label=kind)
    plt.xlabel("Hidden size")
    plt.ylabel("HAR test accuracy")
    plt.title("Figure 4. Hidden-size sweep")
    plt.legend()
    plt.tight_layout()
    FIGURES_1.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_1 / "p1_hidden_size_sweep.png", dpi=160)
    plt.close()
    return rows


def run_clipping_ablation(fast=FAST_MODE):
    loaders = build_adding_loaders(200, fast=fast)
    rows = []
    for clipping in (None, 1.0):
        result = train_regressor(
            RecurrentRegressor(2, SETTINGS.hidden_size, "RNN"), *loaders,
            1 if fast else SETTINGS.epochs, SETTINGS.lr, device(),
            SETTINGS.seeds[0], clip=clipping,
        )
        rows.append((clipping or "none", result["test_mse"]))
    plt.figure(figsize=(5, 3))
    plt.bar([str(x[0]) for x in rows], [x[1] for x in rows])
    plt.axhline(0.167, color="red", linestyle="--", label="constant baseline")
    plt.ylabel("Test MSE")
    plt.title("Figure 5. RNN gradient-clipping ablation at T=200")
    plt.legend()
    plt.tight_layout()
    FIGURES_1.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_1 / "p1_clipping_ablation.png", dpi=160)
    plt.close()
    return rows


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
    FIGURES_1.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for kind in ("RNN", "LSTM", "GRU"):
        points = [(T, result["test_mse"]) for T, k, _, result in rows if k == kind]
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=kind)
    plt.axhline(0.167, color="red", linestyle="--", label="constant baseline")
    plt.axhline(0.05, color="green", linestyle=":", label="solved threshold")
    plt.xlabel("Sequence length")
    plt.ylabel("Test MSE")
    plt.title("Figure 6. Adding-problem MSE against sequence length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_1 / "p1_mse_vs_length.png", dpi=160)
    plt.close()
    solved = {}
    for T, kind, _, result in rows:
        solved.setdefault((kind, T), 0)
        solved[(kind, T)] += int(result["test_mse"] < 0.05)
    with (ROOT / "problem1" / "adding_solved_runs.csv").open("w", encoding="utf-8") as handle:
        handle.write("model,seq_len,solved_runs,total_runs\n")
        for (kind, T), count in sorted(solved.items()):
            handle.write(f"{kind},{T},{count},3\n")
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

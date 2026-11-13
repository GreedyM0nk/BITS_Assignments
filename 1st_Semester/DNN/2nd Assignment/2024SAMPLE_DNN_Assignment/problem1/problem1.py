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
from src.results import append_result, clear_results
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
    clear_results(RESULTS, {"P1-HAR"})
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
    # cuDNN RNN backward requires training mode; no optimizer step occurs here.
    model.train()
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
    rows = []
    for seed in SETTINGS.seeds:
        train, valid, test = har_loaders(seed)
        for size in sizes:
            for kind in ("LSTM", "GRU"):
                result = train_classifier(
                    RecurrentClassifier(9, size, 6, kind), train, valid, test,
                    1 if fast else SETTINGS.epochs, SETTINGS.lr, device(), seed,
                )
                rows.append((size, kind, seed, result["accuracy"], result["params"]))
    plt.figure(figsize=(6, 4))
    for kind in ("LSTM", "GRU"):
        means = []
        errors = []
        for size in sizes:
            values = [accuracy for s, k, _, accuracy, _ in rows
                      if s == size and k == kind]
            means.append(np.mean(values))
            errors.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        plt.errorbar(sizes, means, yerr=errors, marker="o", capsize=3, label=kind)
    with (ROOT / "problem1" / "hidden_size_sweep.csv").open("w", encoding="utf-8") as handle:
        handle.write("hidden_size,model,seed,test_accuracy,params\n")
        handle.writelines(
            f"{size},{kind},{seed},{accuracy},{params}\n"
            for size, kind, seed, accuracy, params in rows
        )
    plt.xlabel("Hidden size")
    plt.ylabel("HAR test accuracy (mean +/- sample std)")
    plt.title("Figure 4. Hidden-size sweep")
    plt.legend()
    plt.tight_layout()
    FIGURES_1.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_1 / "p1_hidden_size_sweep.png", dpi=160)
    plt.close()
    return rows


def run_clipping_ablation(fast=FAST_MODE):
    rows = []
    for seed in SETTINGS.seeds:
        loaders = build_adding_loaders(200, fast=fast)
        for clipping in (None, 1.0):
            result = train_regressor(
                RecurrentRegressor(2, SETTINGS.hidden_size, "RNN"), *loaders,
                1 if fast else SETTINGS.epochs, SETTINGS.lr, device(),
                seed, clip=clipping,
            )
            rows.append((clipping or "none", seed, result["test_mse"]))
    plt.figure(figsize=(5, 3))
    labels = ("none", "1.0")
    grouped = {
        name: [mse for label, _, mse in rows if str(label) == name]
        for name in labels
    }
    means = [np.mean(grouped[name]) for name in labels]
    errors = [np.std(grouped[name], ddof=1) for name in labels]
    plt.bar(labels, means, yerr=errors, capsize=3)
    plt.axhline(0.167, color="red", linestyle="--", label="constant baseline")
    plt.ylabel("Test MSE")
    plt.title("Figure 5. RNN gradient-clipping ablation at T=200")
    plt.legend()
    plt.tight_layout()
    FIGURES_1.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_1 / "p1_clipping_ablation.png", dpi=160)
    plt.close()
    with (ROOT / "problem1" / "clipping_ablation.csv").open("w", encoding="utf-8") as handle:
        handle.write("clipping,seed,test_mse\n")
        handle.writelines(f"{label},{seed},{mse}\n" for label, seed, mse in rows)
    return rows


def run_forget_bias_ablation(fast=FAST_MODE):
    """Compare LSTM forget-gate initialisation at T=200.

    This is kept separate from the selected clipping ablation because the
    instructor clarification withdrew ID-derived variants.
    """
    rows = []
    for seed in SETTINGS.seeds:
        loaders = build_adding_loaders(200, fast=fast)
        for forget_bias in (0.0, 1.0):
            model = RecurrentRegressor(2, SETTINGS.hidden_size, "LSTM")
            with torch.no_grad():
                hidden = SETTINGS.hidden_size
                model.rnn.bias_ih_l0[hidden:2 * hidden].fill_(forget_bias)
                model.rnn.bias_hh_l0[hidden:2 * hidden].zero_()
            result = train_regressor(
                model, *loaders, 1 if fast else SETTINGS.epochs, SETTINGS.lr,
                device(), seed, clip=1.0,
            )
            rows.append((forget_bias, seed, result["test_mse"]))
    with (ROOT / "problem1" / "forget_bias_ablation.csv").open("w", encoding="utf-8") as handle:
        handle.write("forget_bias,seed,test_mse\n")
        handle.writelines(f"{bias},{seed},{mse}\n" for bias, seed, mse in rows)
    return rows


def run_adding(fast=FAST_MODE):
    clear_results(RESULTS, {"P1-adding"})
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
                    "notes": f"mode={'fast' if fast else 'full'};baseline=0.167;solved={result['test_mse'] < 0.05}",
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
    t200 = [row for row in rows if row[0] == 200 and row[2] == SETTINGS.seeds[0]]
    if t200:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for T, kind, _, result in t200:
            history = np.asarray(result["history"])
            axes[0].plot(history[:, 0], label=kind)
            axes[1].plot(history[:, 1], label=kind)
        axes[0].set_title("Figure 7. T=200 training curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training MSE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation MSE")
        for axis in axes:
            axis.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_1 / "p1_adding_T200_learning_curves.png", dpi=160)
        plt.close(fig)
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

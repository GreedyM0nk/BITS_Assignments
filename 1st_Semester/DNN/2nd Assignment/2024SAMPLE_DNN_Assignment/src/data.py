import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def make_adding(n: int, T: int, seed: int):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T, 2), dtype=np.float32)
    x[:, :, 0] = rng.random((n, T))
    i1 = rng.integers(0, T // 2, n)
    i2 = rng.integers(T // 2, T, n)
    x[np.arange(n), i1, 1] = 1
    x[np.arange(n), i2, 1] = 1
    y = x[np.arange(n), i1, 0] + x[np.arange(n), i2, 0]
    return x, y.astype(np.float32)


def validate_adding(x, y):
    marker_count = x[:, :, 1].sum(axis=1)
    if not np.all(marker_count == 2):
        raise AssertionError("each adding sequence must contain two markers")
    midpoint = x.shape[1] // 2
    if not np.all(x[:, :midpoint, 1].sum(1) == 1) or not np.all(x[:, midpoint:, 1].sum(1) == 1):
        raise AssertionError("markers must be split between the two halves")
    values = (x[:, :, 0] * x[:, :, 1]).sum(1)
    np.testing.assert_allclose(values, y, atol=1e-6)


def _read_signal(root: Path, split: str, name: str):
    return np.loadtxt(root / split / "Inertial Signals" / f"{name}_{split}.txt")


def download_har(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "UCI HAR Dataset"
    if extracted.exists():
        return extracted
    archive = data_dir / "har.zip"
    urllib.request.urlretrieve(HAR_URL, archive)
    with zipfile.ZipFile(archive) as outer:
        outer.extractall(data_dir)
    nested = data_dir / "UCI HAR Dataset.zip"
    if nested.exists():
        with zipfile.ZipFile(nested) as inner:
            inner.extractall(data_dir)
    return extracted


def load_har(data_dir: Path, validation_seed: int = 42):
    root = download_har(data_dir)
    channels = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x",
                "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
    train = np.stack([_read_signal(root, "train", name) for name in channels], axis=-1)
    test = np.stack([_read_signal(root, "test", name) for name in channels], axis=-1)
    y_train = np.loadtxt(root / "train" / "y_train.txt", dtype=np.int64) - 1
    y_test = np.loadtxt(root / "test" / "y_test.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(root / "train" / "subject_train.txt", dtype=np.int64)
    test_subjects = np.loadtxt(root / "test" / "subject_test.txt", dtype=np.int64)
    rng = np.random.default_rng(validation_seed)
    validation_subjects = rng.choice(np.unique(subjects), 4, replace=False)
    valid_mask = np.isin(subjects, validation_subjects)
    fit_mask = ~valid_mask
    mean = train[fit_mask].reshape(-1, train.shape[-1]).mean(0)
    std = train[fit_mask].reshape(-1, train.shape[-1]).std(0)
    std = np.maximum(std, 1e-6)
    normalize = lambda a: ((a - mean) / std).astype(np.float32)
    return (normalize(train[fit_mask]), y_train[fit_mask],
            normalize(train[valid_mask]), y_train[valid_mask],
            normalize(test), y_test, subjects[fit_mask], test_subjects, validation_subjects)


def loaders_from_arrays(x, y, batch_size=128, shuffle=True):
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=batch_size, shuffle=shuffle)


def download_shakespeare(data_dir: Path) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "tinyshakespeare.txt"
    if not path.exists():
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


def character_data(text: str, context: int = 128):
    chars = sorted(set(text))
    encode = {c: i for i, c in enumerate(chars)}
    values = np.array([encode[c] for c in text], dtype=np.int64)
    split = int(len(values) * 0.9)
    train, valid = values[:split], values[split:]

    def windows(array):
        count = len(array) - context
        x = np.stack([array[i:i + context] for i in range(count)])
        y = np.stack([array[i + 1:i + context + 1] for i in range(count)])
        return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

    return windows(train), windows(valid), chars, encode

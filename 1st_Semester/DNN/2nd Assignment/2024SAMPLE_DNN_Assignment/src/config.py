from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES_1 = ROOT / "problem1" / "figures"
FIGURES_2 = ROOT / "problem2" / "figures"
RESULTS = ROOT / "results.csv"


@dataclass(frozen=True)
class Settings:
    hidden_size: int = 64
    transformer_dim: int = 128
    heads: int = 4
    layers: int = 4
    context: int = 128
    seeds: tuple[int, ...] = (42, 43, 44)
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 12


SETTINGS = Settings()
RESULT_COLUMNS = [
    "problem", "model", "seed", "seq_len", "test_acc", "macro_f1",
    "test_mse", "params", "sec_per_epoch", "val_loss_nats",
    "bits_per_char", "notes",
]

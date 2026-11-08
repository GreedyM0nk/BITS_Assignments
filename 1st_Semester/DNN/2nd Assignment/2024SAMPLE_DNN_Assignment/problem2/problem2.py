"""Problem 2 runner and Transformer evidence utilities."""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import FIGURES_2, RESULTS, SETTINGS
from src.data import character_data, download_shakespeare, load_har, loaders_from_arrays
from src.models import CausalTransformerLM, EncoderBlock, MyMultiHeadAttention, scaled_dot_product_attention
from src.reproducibility import device, seed_everything
from src.results import append_result
from src.training import count_parameters


def hand_check_attention():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    out, weights = scaled_dot_product_attention(q, q, q)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 3), atol=1e-6)
    reference = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    assert torch.allclose(out, reference, atol=1e-5)
    return out, weights


def plot_hand_check(path):
    _, weights = hand_check_attention()
    FIGURES_2.mkdir(parents=True, exist_ok=True)
    plt.imshow(weights[0].numpy(), cmap="viridis")
    plt.colorbar(label="attention weight")
    plt.xlabel("key position")
    plt.ylabel("query position")
    plt.title("Figure 1. Hand-checkable 3-token attention")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_language_model(fast=True):
    text = download_shakespeare(ROOT / "data")
    context = 64 if fast else SETTINGS.context
    train_set, valid_set, chars, encode = character_data(text, context)
    batch = 32 if fast else SETTINGS.batch_size
    model = CausalTransformerLM(len(chars), SETTINGS.transformer_dim, SETTINGS.heads,
                                2 if fast else SETTINGS.layers, context).to(device())
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS.lr)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch, shuffle=False)
    epochs = 1 if fast else SETTINGS.epochs
    history = []
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device()), y.to(device())
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, len(chars)), y.reshape(-1))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = []
            for x, y in valid_loader:
                x, y = x.to(device()), y.to(device())
                losses.append(torch.nn.functional.cross_entropy(
                    model(x).reshape(-1, len(chars)), y.reshape(-1)).item())
        history.append(sum(losses) / len(losses))
    val_loss = history[-1]
    append_result(RESULTS, {
        "problem": "P2-LM", "model": "CausalTransformer", "seed": SETTINGS.seeds[0],
        "params": count_parameters(model), "val_loss_nats": val_loss,
        "bits_per_char": val_loss / math.log(2),
        "notes": f"context={context};heads={SETTINGS.heads}",
    })
    FIGURES_2.mkdir(parents=True, exist_ok=True)
    plt.plot(history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss (nats)")
    plt.title("Figure 2. Tiny Shakespeare validation learning curve")
    plt.tight_layout()
    plt.savefig(FIGURES_2 / "p2_lm_learning_curve.png", dpi=160)
    plt.close()
    return model, chars, encode, history


def generate_samples(model, chars, encode, path):
    seed_everything(SETTINGS.seeds[0])
    inverse = {value: key for key, value in encode.items()}
    start = torch.tensor([[encode["\n"]]], device=device())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for temperature in (0.2, 0.8, 1.5):
            sample = model.generate(start, 500, temperature=temperature, top_k=20)[0]
            handle.write(f"Temperature {temperature}\n")
            handle.write("".join(inverse[int(i)] for i in sample.cpu()) + "\n\n")


if __name__ == "__main__":
    out, weights = hand_check_attention()
    print("attention checks passed", tuple(out.shape), tuple(weights.shape))
    if "--full" in sys.argv:
        model, chars, encode, _ = train_language_model(False)
        generate_samples(model, chars, encode, ROOT / "problem2" / "generated_samples.txt")

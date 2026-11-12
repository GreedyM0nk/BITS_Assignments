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
from src.models import (
    CausalTransformerLM, EncoderBlock, HARTransformerClassifier,
    MyMultiHeadAttention, scaled_dot_product_attention,
)
from src.reproducibility import device, seed_everything
from src.results import append_result, clear_results
from src.training import count_parameters


def hand_check_attention():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    k = torch.tensor([[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    out, weights = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 3), atol=1e-6)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, reference, atol=1e-5)
    causal = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    _, causal_weights = scaled_dot_product_attention(q, k, v, causal)
    assert torch.equal(
        causal_weights.masked_select(~causal),
        torch.zeros_like(causal_weights.masked_select(~causal)),
    )
    return out, weights


def hand_attention_table():
    """Return hand-computed scores, weights, and output for explicit Q/K/V."""
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    k = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scores = q @ k.T / math.sqrt(2.0)
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return q, k, v, scores, weights, output


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


def run_har_transformer(fast=True):
    clear_results(RESULTS, {"P2-HAR"})
    results = []
    for seed in SETTINGS.seeds:
        arrays = load_har(ROOT / "data", seed)
        x_train, y_train, x_valid, y_valid, x_test, y_test = arrays[:6]
        loaders = (
            loaders_from_arrays(x_train, y_train),
            loaders_from_arrays(x_valid, y_valid, shuffle=False),
            loaders_from_arrays(x_test, y_test, shuffle=False),
        )
        for positional in ("none", "sinusoidal", "learned"):
            model = HARTransformerClassifier(positional=positional).to(device())
            optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS.lr)
            criterion = torch.nn.CrossEntropyLoss()
            epochs = 1 if fast else SETTINGS.epochs
            for _ in range(epochs):
                model.train()
                for x, y in loaders[0]:
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(x.to(device())), y.to(device()))
                    loss.backward()
                    optimizer.step()
            model.eval()
            truth, predictions = [], []
            with torch.no_grad():
                for x, y in loaders[2]:
                    predictions.extend(model(x.to(device())).argmax(-1).cpu().tolist())
                    truth.extend(y.tolist())
            accuracy = sum(a == b for a, b in zip(truth, predictions)) / len(truth)
            append_result(RESULTS, {
                "problem": "P2-HAR", "model": f"Transformer-{positional}",
                "seed": seed, "test_acc": accuracy,
                "params": count_parameters(model), "notes": "CLS;positional ablation",
            })
            results.append((positional, seed, accuracy, count_parameters(model)))
    with (ROOT / "problem2" / "transformer_har_results.csv").open("w", encoding="utf-8") as handle:
        handle.write("positional,seed,test_accuracy,params\n")
        handle.writelines(
            f"{positional},{seed},{accuracy},{params}\n"
            for positional, seed, accuracy, params in results
        )
    return results


def plot_attention_heatmaps(model, tokens, path):
    model.eval()
    x = model.embedding(tokens)
    x = model.position(x)
    _, weights = model.blocks[0](x, return_weights=True)
    FIGURES_2.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for index, axis in enumerate(axes):
        axis.imshow(weights[0, index].detach().cpu(), cmap="magma")
        axis.set_title(f"Head {index + 1}")
        axis.set_xlabel("key position")
        axis.set_ylabel("query position")
    fig.suptitle("Figure 3. Transformer attention heads")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_transformer_ablations(fast=True):
    widths = (1, 4) if fast else (1, 4, 8)
    rows = []
    for heads in widths:
        model = CausalTransformerLM(
            65, SETTINGS.transformer_dim, heads, 1, 128
        ).to(device())
        tokens = torch.randint(0, 65, (2, 128), device=device())
        model.eval()
        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start:
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            with torch.no_grad():
                model(tokens)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000
        else:
            import time
            begin = time.perf_counter()
            with torch.no_grad():
                model(tokens)
            elapsed = time.perf_counter() - begin
        rows.append((heads, elapsed, count_parameters(model)))
    with (ROOT / "problem2" / "transformer_ablations.csv").open("w", encoding="utf-8") as handle:
        handle.write("heads,seconds,params\n")
        handle.writelines(f"{h},{seconds},{params}\n" for h, seconds, params in rows)
    return rows


def run_mask_and_context_ablation(fast=True):
    contexts = (64, 128) if fast else (64, 128, 256, 512)
    rows = []
    import time
    for context in contexts:
        model = CausalTransformerLM(
            65, SETTINGS.transformer_dim, SETTINGS.heads, 1, context
        ).to(device())
        tokens = torch.randint(0, 65, (2, context), device=device())
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            causal_logits = model(tokens, causal=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        causal_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with torch.no_grad():
            unmasked_logits = model(tokens, causal=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        unmasked_seconds = time.perf_counter() - started
        peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        rows.append((context, causal_seconds, unmasked_seconds, peak_memory,
                     float(causal_logits.mean().detach()),
                     float(unmasked_logits.mean().detach())))
    with (ROOT / "problem2" / "context_ablation.csv").open("w", encoding="utf-8") as handle:
        handle.write("context,causal_seconds,unmasked_seconds,peak_memory,causal_mean,unmasked_mean\n")
        handle.writelines(",".join(map(str, row)) + "\n" for row in rows)
    return rows


def train_language_model(fast=True):
    clear_results(RESULTS, {"P2-LM"})
    text = download_shakespeare(ROOT / "data")
    context = 64 if fast else SETTINGS.context
    train_set, valid_set, chars, encode = character_data(text, context, stride=context // 2)
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
            loss_total, token_total = 0.0, 0
            for x, y in valid_loader:
                x, y = x.to(device()), y.to(device())
                token_count = y.numel()
                loss_total += torch.nn.functional.cross_entropy(
                    model(x).reshape(-1, len(chars)), y.reshape(-1), reduction="sum").item()
                token_total += token_count
        history.append(loss_total / token_total)
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
            sample = model.generate(start, 500, temperature=temperature, top_k=20)[0][1:]
            handle.write(f"Temperature {temperature}\n")
            handle.write("".join(inverse[int(i)] for i in sample.cpu()) + "\n\n")


if __name__ == "__main__":
    out, weights = hand_check_attention()
    print("attention checks passed", tuple(out.shape), tuple(weights.shape))
    if "--full" in sys.argv:
        model, chars, encode, _ = train_language_model(False)
        generate_samples(model, chars, encode, ROOT / "problem2" / "generated_samples.txt")

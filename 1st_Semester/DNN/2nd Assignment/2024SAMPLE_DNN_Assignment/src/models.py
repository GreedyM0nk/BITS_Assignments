import math
from typing import Optional

import torch
from torch import nn


class MyRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x, h):
        return torch.tanh(
            torch.nn.functional.linear(x, self.weight_ih, self.bias_ih)
            + torch.nn.functional.linear(h, self.weight_hh, self.bias_hh)
        )


class MyLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x, state):
        h, c = state
        gates = (
            torch.nn.functional.linear(x, self.weight_ih, self.bias_ih)
            + torch.nn.functional.linear(h, self.weight_hh, self.bias_hh)
        )
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = i.sigmoid(), f.sigmoid(), o.sigmoid()
        c_next = f * c + i * g.tanh()
        return o * c_next.tanh(), c_next


class MyGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x, h):
        x_gates = torch.nn.functional.linear(x, self.weight_ih, self.bias_ih)
        h_gates = torch.nn.functional.linear(h, self.weight_hh, self.bias_hh)
        x_r, x_z, x_n = x_gates.chunk(3, dim=-1)
        h_r, h_z, h_n = h_gates.chunk(3, dim=-1)
        r = (x_r + h_r).sigmoid()
        z = (x_z + h_z).sigmoid()
        n = (x_n + r * h_n).tanh()
        return (1 - z) * n + z * h


class RecurrentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, classes, kind="RNN"):
        super().__init__()
        self.kind = kind.upper()
        layer = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[self.kind]
        self.rnn = layer(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, classes)

    def forward(self, x):
        output, state = self.rnn(x)
        h = state[0] if self.kind == "LSTM" else state
        return self.head(h[-1])


class RecurrentRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, kind="RNN"):
        super().__init__()
        layer = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind.upper()]
        self.rnn = layer(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, state = self.rnn(x)
        h = state[0] if isinstance(state, tuple) else state
        return self.head(h[-1]).squeeze(-1)


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


class SinusoidalEncoding(nn.Module):
    def __init__(self, width, max_len=4096):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        divisor = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000) / width))
        pe = torch.zeros(max_len, width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * divisor), torch.cos(position * divisor)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MyMultiHeadAttention(nn.Module):
    def __init__(self, width, heads, dropout=0.0):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width, self.heads, self.head_width = width, heads, width // heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.dropout = dropout

    def forward(self, x, mask=None, return_weights=False):
        batch, steps, _ = x.shape
        shape = (batch, steps, self.heads, self.head_width)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        out, weights = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch, steps, self.width)
        result = self.out_proj(out)
        return (result, weights) if return_weights else result


class EncoderBlock(nn.Module):
    def __init__(self, width, heads, dropout=0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = MyMultiHeadAttention(width, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(4 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_weights=False):
        attended = self.attn(self.norm1(x), mask, return_weights)
        if return_weights:
            attended, weights = attended
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return (x, weights) if return_weights else x


class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size, width=128, heads=4, layers=4, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = SinusoidalEncoding(width, max_len)
        self.blocks = nn.ModuleList([EncoderBlock(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size)

    def forward(self, tokens):
        steps = tokens.size(1)
        causal = torch.tril(torch.ones(steps, steps, device=tokens.device, dtype=torch.bool))
        x = self.position(self.embedding(tokens))
        for block in self.blocks:
            x = block(x, causal)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, tokens, length, temperature=0.8, top_k=20):
        for _ in range(length):
            logits = self(tokens[:, -self.position.pe.size(1):])[:, -1] / temperature
            values, indices = logits.topk(min(top_k, logits.size(-1)))
            probs = torch.softmax(values, dim=-1)
            next_token = indices.gather(-1, torch.multinomial(probs, 1))
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

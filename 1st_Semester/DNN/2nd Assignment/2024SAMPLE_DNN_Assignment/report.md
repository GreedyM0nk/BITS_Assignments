# Deep Neural Networks - Programming Assignment 2
## RNN, LSTM, GRU and Transformer

**Name:** Souvik Dutta
**Student ID:** 2026AJ05991
**Configuration:** Problem 1 `h=64`; Problem 2 `d=128`, 4 heads; seeds
`42, 43, 44`. The individual ablation is vanilla-RNN gradient clipping at
`T=200`, norm `1.0`, versus no clipping.

> **Submission rule:** replace every bracketed value below with a value from
> the saved Colab CSVs/notebooks. Do not copy values from the sample report.
> Keep the final report to six A4 pages or fewer.

## 1. Problem 1 - recurrent models

### 1.1 Setup and verification

UCI HAR uses nine inertial channels, 128 time steps, and six activities. The
split is subject-wise: validation subjects are selected from the training
subjects for each seed, and normalization statistics are fitted on the
remaining training subjects only. Figure 1 shows the class balance. The
adding task uses the required two marker positions and target sum; the direct
assert cell and `validate_adding` check are saved in the notebook.

The custom cells use the standard equations with PyTorch-compatible gate
ordering. Full-sequence tests compare `MyRNNCell`, `MyLSTMCell`, and
`MyGRUCell` with `nn.RNN`, `nn.LSTM`, and `nn.GRU`; the maximum absolute
differences are recorded in the notebook output. For input size 9 and
`h=64`, the one-bias equation counts are

`RNN = h*d + h*h + h = 4,736`,
`LSTM = 4*4,736 = 18,944`,
`GRU = 3*4,736 = 14,208`.

PyTorch stores separate input-hidden and hidden-hidden biases, giving
`4,800`, `19,200`, and `14,400`, respectively.

### 1.2 HAR classification

**Table 1.** Fill from `results.csv`, grouped by model and seed; report
mean +/- sample standard deviation for test accuracy, macro-F1, parameters,
and seconds per epoch.

| Model | Accuracy (%) | Macro-F1 (%) | Parameters | s/epoch |
|---|---:|---:|---:|---:|
| RNN | [ ] | [ ] | [ ] | [ ] |
| LSTM | [ ] | [ ] | [ ] | [ ] |
| GRU | [ ] | [ ] | [ ] | [ ] |

Discuss the best model and the largest confusion categories using Figure 2.

### 1.3 Adding problem

**Table 2.** Fill from `results.csv` and `adding_solved_runs.csv`. The
constant-prediction baseline is approximately `0.167`; a solved run has
test MSE `< 0.05`.

| T | RNN MSE +/- std | solved | LSTM MSE +/- std | solved | GRU MSE +/- std | solved |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | [ ] | [ ]/3 | [ ] | [ ]/3 | [ ] | [ ]/3 |
| 100 | [ ] | [ ]/3 | [ ] | [ ]/3 | [ ] | [ ]/3 |
| 200 | [ ] | [ ]/3 | [ ] | [ ]/3 | [ ] | [ ]/3 |
| 400 | [ ] | [ ]/3 | [ ] | [ ]/3 | [ ] | [ ]/3 |

Figure 3 shows the MSE-length trend and Figure 4 shows the `T=200`
train/validation curves. Explain whether each model learns the markers or
collapses to the baseline.

### 1.4 Gradients, capacity, and ablation

Figure 5 reports before/after gradient norms at `T=100`. Figure 6 reports the
three-seed hidden-size sweep for `h` in `{16,32,64,128}`. Figure 7 reports the
three-seed clipping ablation at `T=200`. Interpret the plots in terms of
vanishing gradients, parameter cost, and convergence rather than making an
unsupported claim.

For a wearable recommendation, select the model using the measured accuracy,
seed spread, parameter count, runtime, and long-sequence behavior. State the
chosen model and cite the corresponding table values.

## 2. Problem 2 - Transformer from scratch

### 2.1 Attention and encoder block

The implementation computes `softmax(QK^T/sqrt(d_k))V` and applies boolean
causal masks before softmax. For the required hand check:

```
Q = [[1,0], [0,1], [1,1]]
K = [[1,1], [1,0], [0,1]]
V = [[1,2], [3,4], [5,6]]
```

With `a=1/sqrt(2)=0.7071`, the unmasked score matrix is
`[[a,a,0],[a,0,a],[2a,a,a]]`. The code output in the notebook is the
authoritative numerical result and is checked against the hand calculation
and PyTorch's scaled-dot-product attention. The causal check asserts exact
zero weight above the diagonal.

The encoder block is pre-LayerNorm with Q/K/V/O projections and a
`d -> 4d -> d` GELU feed-forward network. For `d=128`,
`4d^2 + 2*d*(4d) = 196,608` weights; the notebook records biases,
LayerNorm parameters, and the total.

### 2.2 HAR Transformer

**Table 3.** Fill from `transformer_har_results.csv`, reporting mean +/- sample
standard deviation over seeds.

| Positional mode | Accuracy (%) | Macro-F1 (%) | Parameters |
|---|---:|---:|---:|
| None | [ ] | [ ] | [ ] |
| Sinusoidal | [ ] | [ ] | [ ] |
| Learned | [ ] | [ ] | [ ] |

Compare the best Transformer with the best recurrent model using the same
seeds and discuss parameter/runtime trade-offs.

### 2.3 Character language model

The causal character model uses context 128, width 128, four heads, sinusoidal
positions, and the configured number of layers. Report the measured validation
loss and bits per character from `results.csv`; include Figure 8 (learning
curve) and the three genuine 500-character samples from
`generated_samples.txt` at temperatures `0.2`, `0.8`, and `1.5`.

### 2.4 Analysis and ablations

Use the saved heatmap figure to identify a local and a long-range head.
Use `transformer_ablations.csv` for head count, `context_ablation.csv` for
context length and memory, and the notebook output for causal versus
unmasked behavior. Explain the `O(n^2 d)` attention-memory term and do not
describe an unmasked training result as a valid causal language model.

## 3. AI-use statement

AI assistance was used for code structuring, debugging, test design, plotting
helpers, and report organization. The implementation was checked with the
repository tests, and the reported experiment values were generated by the
author's Colab T4 run and reconciled against the saved CSVs and notebook
outputs. Any limitation or interrupted run is stated explicitly; no
unexecuted result is presented as measured evidence.

## References

1. Hochreiter and Schmidhuber, “Long Short-Term Memory,” 1997.
2. Cho et al., “Learning Phrase Representations using RNN Encoder-Decoder,”
   2014.
3. Vaswani et al., “Attention Is All You Need,” 2017.
4. Anguita et al., UCI Human Activity Recognition Using Smartphones, 2013.
5. Karpathy, Tiny Shakespeare dataset.
6. PyTorch documentation for recurrent layers, attention, and optimizers.

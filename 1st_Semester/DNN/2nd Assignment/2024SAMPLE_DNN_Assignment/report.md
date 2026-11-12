# Deep Neural Networks - Programming Assignment 2
## RNN, LSTM, GRU and Transformer

**Name:** Souvik Dutta
**Student ID:** 2026AJ05991
**Configuration:** `h=64`; Transformer `d=128`, 4 heads; seeds `42, 43, 44`.
The individual ablation is vanilla-RNN gradient clipping at `T=200`, norm
`1.0`, versus no clipping. The instructor clarification withdrawing the
variants table was followed.

## 1. Problem 1

UCI HAR uses nine inertial channels, 128 time steps, six activities, and a
subject-wise split. Normalisation statistics are fitted only on the training
subjects. The adding task uses two marker positions and the target sum. The
custom cells were checked over complete sequences against PyTorch layers.
For input size 9 and `h=64`, one-bias equation counts are RNN 4,736, LSTM
18,944, and GRU 14,208; PyTorch's two bias vectors give 4,800, 19,200, and
14,400.

### HAR classification

| Model | Accuracy mean +/- sd | Macro-F1 mean +/- sd | Parameters | s/epoch mean |
|---|---:|---:|---:|---:|
| RNN | 65.739 +/- 3.335% | 64.994 +/- 3.338% | 5,190 | 0.186 |
| LSTM | 85.997 +/- 2.952% | 85.664 +/- 3.108% | 19,590 | 0.183 |
| GRU | **88.847 +/- 0.674%** | **88.838 +/- 0.731%** | 14,790 | 0.176 |

The GRU is the strongest wearable choice: it has the best accuracy and
macro-F1, fewer parameters than the LSTM, and the lowest measured epoch time.
The confusion matrix and class-balance plot are included in `problem1/figures`.

### Adding problem

Values below are means over the three seeds; `solved` counts runs with MSE
below 0.05. The constant predictor baseline is approximately 0.167.

| T | RNN MSE +/- sd | solved | LSTM MSE +/- sd | solved | GRU MSE +/- sd | solved |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.17459 +/- 0.00043 | 0/3 | 0.17044 +/- 0.00287 | 0/3 | **0.00965 +/- 0.00178** | 3/3 |
| 100 | 0.16973 +/- 0.00028 | 0/3 | 0.16955 +/- 0.00042 | 0/3 | 0.13319 +/- 0.03167 | 0/3 |
| 200 | 0.16812 +/- 0.00025 | 0/3 | 0.16833 +/- 0.00057 | 0/3 | 0.16721 +/- 0.00211 | 0/3 |
| 400 | 0.15367 +/- 0.00070 | 0/3 | 0.15337 +/- 0.00025 | 0/3 | 0.15264 +/- 0.00046 | 0/3 |

The GRU solves the shortest sequence but all models approach the baseline at
longer horizons under the fixed 12-epoch budget. This is consistent with
vanishing-gradient difficulty rather than evidence that the task is solved at
all lengths.

### Gradients, capacity and clipping

The `T=100` gradient figure records before/after profiles for all three
models. The hidden-size sweep used sizes 16, 32, 64 and 128 over all three
seeds. At `h=64`, mean HAR accuracy was 85.997% for LSTM and 88.847% for
GRU; the corresponding parameter counts were 19,590 and 14,790. Clipping
at norm 1.0 produced MSE `0.16845 +/- 0.00045`, versus
`0.16812 +/- 0.00025` without clipping, so this ablation did not improve the
measured result.

## 2. Problem 2 - Transformer from scratch

The implementation computes `softmax(QK^T/sqrt(d_k))V`, applies boolean causal
masks before softmax, and uses pre-LayerNorm encoder blocks. The hand example
uses:

```
Q = [[1,0], [0,1], [1,1]]
K = [[1,1], [1,0], [0,1]]
V = [[1,2], [3,4], [5,6]]
```

For `a=1/sqrt(2)=0.707107`, the score matrix is
`[[a,a,0],[a,0,a],[2a,a,a]]`. Thus the first-row weights are
`[exp(a), exp(a), 1] / [2exp(a)+1] = [0.4011, 0.4011, 0.1978]`,
the second-row weights are
`[exp(a), 1, exp(a)] / [2exp(a)+1] = [0.4011, 0.1978, 0.4011]`,
and the third-row weights are
`[exp(2a), exp(a), exp(a)] / [exp(2a)+2exp(a)]`.
Multiplying each weight row by V gives
`[[2.593327,3.593327],[3.000000,4.000000],[2.489531,3.489531]]`,
matching the code and PyTorch's scaled-dot-product attention.
The causal test verifies exact zero probability above the diagonal.

### Transformer HAR

| Positional mode | Accuracy mean +/- sd | Macro-F1 mean +/- sd | Parameters |
|---|---:|---:|---:|
| None | 87.659 +/- 1.936% | 87.446 +/- 1.980% | 415,494 |
| Sinusoidal | **89.445 +/- 1.240%** | **89.247 +/- 1.303%** | 415,494 |
| Learned | 88.439 +/- 0.859% | 87.984 +/- 0.989% | 415,494 |

The sinusoidal Transformer is competitive with the GRU but is much larger,
so the GRU remains preferable for a constrained wearable deployment.

### Character language model and ablations

The causal language model used context 128, width 128, four heads, four
layers, and 12 epochs. It has 810,049 parameters. Validation loss decreased
from 2.3461 to **1.5941 nats**, or **2.2998 bits per character**. Genuine
samples at temperatures 0.2, 0.8 and 1.5 are saved in
`problem2/generated_samples.txt`; the learning curve and two-head heatmap
are in `problem2/figures`.

The head-count study compared 1, 4 and 8 heads at fixed width; parameter
count remained 215,233 because it uses one Transformer block, whereas the
four-block language model has 810,049 parameters. The encoder-block
derivation is `4d² + 2*d*d_ff`; for `d=128`, `d_ff=512`, the weight-only
term is 196,608, and the notebook compares the complete analytic total with
`sum(p.numel())`.

The context study measured causal versus unmasked passes at contexts 64, 128,
256 and 512. The T4 results were respectively:
`(0.001488, 0.001084, 18.93 MB)`,
`(0.001327, 0.001055, 20.14 MB)`,
`(0.001474, 0.001166, 24.37 MB)`, and
`(0.001569, 0.001551, 39.25 MB)` for causal time, unmasked time and peak
memory. The increase is consistent with the `O(n²d)` attention score/memory
term. Unmasked attention is treated only as an ablation, never as a valid
causal language model. The heatmap shows local diagonal attention in one head
and broader earlier-token attention in the other; future positions remain
masked.

## 3. Reproducibility and AI-use statement

The full matrix was executed on a Colab Tesla T4 with PyTorch
`2.14.1+cu130`, using seeds 42, 43 and 44. The repository tests pass
(7 tests), and the generated CSVs/figures were copied from the completed
runtime. AI assistance was used for code structuring, debugging, test design,
plotting helpers and report organisation. All reported experiment values were
generated by the author's T4 run; no unexecuted result is presented as
measured evidence.

## References

1. Hochreiter and Schmidhuber, “Long Short-Term Memory,” 1997.
2. Cho et al., “Learning Phrase Representations using RNN Encoder-Decoder,” 2014.
3. Vaswani et al., “Attention Is All You Need,” 2017.
4. Anguita et al., UCI Human Activity Recognition Using Smartphones, 2013.
5. Karpathy, Tiny Shakespeare dataset.
6. PyTorch documentation for recurrent layers, attention, and optimizers.

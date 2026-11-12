# DNN Programming Assignment 2

Student: **Souvik Dutta**
Student ID: **2026AJ05991**

Chosen configuration (per the instructor clarification): Problem 1 hidden size
`h=64`, Problem 2 model width `d=128` with 4 heads, and seeds `42, 43, 44`.
The individual ablation is gradient clipping for the vanilla RNN at `T=200`;
this choice is used because the variants table and its ID-based instruction
were explicitly withdrawn.

## Environment and run order

```powershell
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121  # T4/CUDA
jupyter notebook
```

Open `problem1/problem1.ipynb` and `problem2/problem2.ipynb`, then use
**Kernel > Restart Kernel and Run All**. Both notebooks use only relative paths.
Set `FAST_MODE = True` for a smoke run; set it to `False` for the rubric
configuration. The first run downloads UCI HAR and Tiny Shakespeare into
`data/`. The data directory, checkpoints, and virtual environment are excluded
from the submission zip.

The full configuration is intended for a T4 GPU and should be budgeted as a
multi-hour reproducibility run because it includes repeated seeds and
ablations. Run the cells separately, in this order:

1. Problem 1: `run_smoke()`, `run_har(False)`, `run_adding(False)`,
   `run_gradient_study(False)`, `run_hidden_size_sweep(False)`, and
   `run_clipping_ablation(False)`.
2. Problem 2: `hand_check_attention()`, `run_har_transformer(False)`,
   `run_transformer_ablations(False)`, `run_mask_and_context_ablation(False)`,
   `train_language_model(False)`, `generate_samples(...)`, and the heatmap
   cell.

The experiment functions write their CSVs and figures into the assignment
folder. Each experiment clears only its own prior rows before starting and
appends each completed run immediately, so an interrupted cell can be rerun
without creating duplicate result rows for unrelated experiments. The
generated samples are written to `problem2/generated_samples.txt`.

## Reproducibility and assumptions

All Python, NumPy and PyTorch seeds are fixed. HAR validation is subject-wise
and statistics are fit on the training subjects only. The selected values are
declared here and in the report title block, and must be retained when
reproducing the results.

Complete `report.md` from the generated tables and figures, then export it as a
maximum-six-page PDF. The assignment's academic-integrity requirement is
preserved in the report's AI-use statement. The current repository contains
verified code and tests; full benchmark outputs must be generated on a T4
before submission.

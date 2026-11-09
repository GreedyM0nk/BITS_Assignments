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
ablations. Every completed run appends to `results.csv`, so an interrupted
session can be resumed. The generated samples are written to
`problem2/generated_samples.txt`.

## Reproducibility and assumptions

All Python, NumPy and PyTorch seeds are fixed. HAR validation is subject-wise
and statistics are fit on the training subjects only. The selected values are
declared here and in the report title block, and must be retained when
reproducing the results.

The report must be completed from the generated tables and figures. The
assignment's academic-integrity requirement is preserved in the report's
AI-use statement. The current repository contains verified smoke evidence; full
benchmark outputs must be generated on a T4 before submission.

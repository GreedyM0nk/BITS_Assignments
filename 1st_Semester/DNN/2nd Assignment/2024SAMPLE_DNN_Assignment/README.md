# DNN Programming Assignment 2

This sample uses the documented fallback variant because no student ID or
individualisation table was supplied: hidden size `64`, Transformer width
`128`, seeds `42, 43, 44`, and the even-ID gradient-clipping ablation.

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

The full configuration is intended for a T4 GPU. Every completed run appends
to `results.csv`, so an interrupted session can be resumed. The generated
samples are written to `problem2/generated_samples.txt`.

## Reproducibility and assumptions

All Python, NumPy and PyTorch seeds are fixed. HAR validation is subject-wise
and statistics are fit on the training subjects only. The sample defaults above
replace the missing variant table; change them in `src/config.py` if an
instructor variant is later provided.

The report must be completed from the generated tables and figures. The
assignment's academic-integrity requirement is preserved in the report's
AI-use statement.

# Report source scaffold

This file is a source scaffold for the required PDF (maximum six A4 pages,
11-point font and at least 2 cm margins). Export the completed report after
the notebooks finish and replace every placeholder with a generated table or
figure reference.

## Title block

**DNN Programming Assignment 2 - RNN, LSTM, GRU and Transformer**  
Name: **Souvik Dutta**  Student ID: **2026AJ05991**

## Problem 1

Current repository evidence status: the adding-problem fast CPU matrix and
gradient/clipping studies have generated real CSV rows and figures. The HAR
matrix did not finish in the available CPU run window, so HAR metrics and
related figures must be generated on the T4 before submission.

Describe the UCI HAR subject-wise split and training-only normalization. Include
the class-balance figure. Explain the custom-cell equations and show the
library-equivalence test tolerance and hand parameter derivations. Report the
three-seed HAR accuracy/F1 table, timing and best confusion matrix. Report all
12 adding-problem model/length combinations with three seeds, baseline 0.167,
solved-run counts and the T=200 curves. Add gradient, hidden-size and the
sample gradient-clipping ablation figures. Finish with the evidence-backed
wearable recommendation.

## Problem 2

The hand-computed attention table and fast head/context/no-mask measurements are
generated in the repository. The language-model run did not finish within the
available CPU run window; therefore `generated_samples.txt` remains explicitly
labelled as pending rather than containing fabricated text.

Include the hand-checkable attention example, unit-test assertions, parameter
derivation, encoder-block details and HAR positional-encoding comparison.
Report the Tiny Shakespeare learning curve, validation loss and bits per
character, then discuss the three generated temperatures. Include two
attention heatmaps, head-count/no-mask/context-length ablations and the
`O(n^2 d)` interpretation.

## AI-use statement

State the tools used, the exact debugging/plotting/coding assistance received,
and which portions were independently verified. Do not claim experiments that
were not run.

## References

List PyTorch, UCI HAR, and Tiny Shakespeare sources with access dates.

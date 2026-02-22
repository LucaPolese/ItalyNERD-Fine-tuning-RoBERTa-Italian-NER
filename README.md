# ItalyNERD: Fine-tuning RoBERTa for Italian Named Entity Recognition

## Overview
**ItalyNERD** is a reproducible NLP project focused on **fine-grained Italian Named Entity Recognition (NER)** using the **MultiNERD** dataset and a **RoBERTa token-classification pipeline**.

The repository currently includes:
- `italyNERD.ipynb`: the executable notebook containing the full experimental workflow.

## What this notebook is for
The notebook is designed to:
1. Build an **Italian-only NER benchmark pipeline** (`lang=it`) on MultiNERD.
2. Fine-tune a Transformer encoder (RoBERTa) for **BIO sequence labeling**.
3. Evaluate performance with **strict entity-level span matching** (via `seqeval`).
4. Produce auditable outputs (metrics, reports, runtime artifacts) used in the accompanying paper/report.

In short, this notebook is not just a demo: it is the **single reproducible source** for the reported Italian NER results.

## Task and label space
The task is token classification with BIO tags over a fine-grained inventory:
- `O` + `B-`/`I-` for 15 entity types:
  - `PER`, `ORG`, `LOC`, `ANIM`, `BIO`, `CEL`, `DIS`, `EVE`, `FOOD`, `INST`, `MEDIA`, `MYTH`, `PLANT`, `TIME`, `VEHI`
- Total labels: **31**

## Core methodology
### 1) Data
- Dataset: `Babelscape/multinerd` (Hugging Face Datasets)
- Focus: Italian subset (`lang=it`)
- Official splits: train / validation / test

### 2) Tokenization and alignment
- Tokenizer: RoBERTa tokenizer with word-aware mapping (`is_split_into_words=True`)
- Alignment strategy:
  - Special tokens → `-100` (ignored by loss)
  - First subword of a word → original label
  - Subsequent subwords → same label, with `B-X` converted to `I-X`

This preserves BIO consistency under subword tokenization.

### 3) Model and training
- Backbone: `roberta-base`
- Head: token classification (`AutoModelForTokenClassification`)
- Reported setup in the project document includes:
  - LR `2e-5`
  - Epochs `3`
  - Effective batch size `16` (batch `4` × grad accumulation `4`)
  - Warmup + linear decay
  - Weight decay, gradient clipping, checkpointing, and optional fp16 on CUDA

### 4) Evaluation
- Primary metrics: entity-level precision / recall / F1 (`seqeval` strict span matching)
- Secondary metrics: token-level accuracy, per-entity report, micro/macro/weighted averages

## Main findings (from project report)
The project reports strong Italian extraction quality under strict span evaluation, with:
- High performance on head classes (notably `LOC`, `ORG`, `PER`)
- A visible **macro vs micro/weighted gap**, consistent with long-tail difficulty in fine-grained NER
- Challenging low-support classes (e.g., `BIO`) behaving as expected in imbalanced settings

## Why this project matters
This work provides an Italian-focused, transparent baseline where:
- preprocessing choices are explicit,
- alignment semantics are documented,
- and claims are tied to executable notebook outputs.

It is useful for:
- Italian information extraction pipelines,
- benchmarking token-classification NER systems,
- and studying long-tail behavior in fine-grained entity typing.

## Reproducibility notes
The project follows a **notebook-faithful reporting principle**:
- report only values produced by executed runs,
- keep preprocessing/evaluation semantics fixed,
- and rely on generated artifacts for traceability.

For full methodological detail, see `italyNERD.tex`.

## How to use
1. Open `italyNERD.ipynb` in Jupyter/VS Code.
2. Install required Python packages used in the notebook (`transformers`, `datasets`, `evaluate`, `seqeval`, `torch`, etc.).
3. Run cells in order to:
   - load/filter data,
   - tokenize and align labels,
   - train/evaluate the model,
   - export metrics/artifacts.

## Citation and references
The bibliography and technical references (Transformers, MultiNERD, CoNLL-style evaluation, etc.) are included in `italyNERD.tex`.

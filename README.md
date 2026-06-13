# ViEC Benchmark

This repository provides the public reference artifacts for **ViEC: A Pilot Benchmark for Vietnamese E-commerce Intent Detection**.

It is intentionally compact. The repository includes the released benchmark datasets and the core model architecture definitions needed to inspect the method described in the paper. It does not include private training recipes, tuned hyperparameters, checkpoints, local runtime paths, notebook logs, or exploratory analysis code.

## Dataset

The benchmark contains two JSON files:

- `data/ECOM.json`: translated Vietnamese e-commerce intent data used as the source-side dataset.
- `data/NECOM.json`: real-world Vietnamese e-commerce utterances used as the target-side dataset.


The shared intent inventory contains 10 labels:

```text
cancel_order
checking
contact_support
modify_order
payment
product_issue
promotion
purchase
return_refund
shipping
```

## Model Code

The `models/` directory contains cleaned architecture components extracted from the experimental notebooks:

- `models/contextual.py`
  - `ContextualFeatureBranch`
  - Projects PhoBERT/XLM-R style transformer hidden states back to token-level features.
  - Uses token-to-subword alignment through `word_positions`.

- `models/hierarchical.py`
  - `BaselineCNN`
  - `HierarchicalCNN`
  - `HierarchicalBiLSTM`
  - `HierarchicalBiGRU`
  - `HierarchicalIntentSlotModel`
  - `CNNJointModel`
  - Defines the main hierarchical intent/slot architecture family used by the paper.

The code is intended for method inspection and extension. It does not hard-code the tuned experimental settings used in the paper.

## Evaluation Protocol

The paper uses a source-to-target benchmark protocol:

1. Train and validate on ECOM using stratified folds.
2. Evaluate each trained fold on the fixed NECOM target set.
3. Report ECOM validation accuracy, NECOM external accuracy, and the generalization gap.

This repository exposes the dataset and model definitions needed to implement that protocol, while leaving training configuration choices to the user.

## Minimal Usage

Install the expected runtime dependencies in your own environment:

```bash
pip install torch transformers
```

Then import the architecture modules directly:

```python
from models.contextual import ContextualFeatureBranch
from models.hierarchical import HierarchicalBiGRU, HierarchicalIntentSlotModel
```

Example dataset loading:

```python
import json
from pathlib import Path

records = json.loads(Path("data/ECOM.json").read_text(encoding="utf-8"))
print(len(records), records[0])
```

## License

This repository is released under the MIT License. See `LICENSE` for details.

# VascGraphROM

VascGraphROM is a cleaned Graph Neural Network workflow for 1D vascular reduced-order modeling.

## Quick Start

### 1) Install dependencies
Use your preferred Python environment and install required packages (PyTorch, DGL, NumPy, PyYAML, tqdm).

### 2) Configure data location
Set the data root in `data_location.txt` (or copy from `data_location_example.txt`).

### 3) Train
```bash
python network1d/train.py --config configs/baseline_mgn.yaml
```

### 4) Evaluate (dataset-level)
```bash
python network1d/eval.py --config configs/baseline_mgn.yaml --ckpt models/xxx/trained_gnn.pms
```

### 5) Rollout evaluation (per-graph)
```bash
python network1d/rollout_eval.py --config configs/baseline_mgn.yaml --ckpt models/xxx/trained_gnn.pms
```

## Project Structure

- `network1d/`: training, inference, rollout, model and pipelines.
- `graph1d/`: graph/data generation and normalization utilities.
- `configs/`: YAML experiment configs (`baseline_mgn.yaml` is the default baseline).
- `tools/`: I/O and plotting helpers.
- `test/`: regression tests and sample artifacts.

## Notes

- `--config` is now standardized across train/eval/rollout commands.
- `--ckpt` should point to the exact `trained_gnn.pms` file.
- Evaluation commands automatically resolve the model directory from `--ckpt`.

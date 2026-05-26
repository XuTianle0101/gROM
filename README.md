# VascGraphROM

VascGraphROM is a Graph Neural Network (GNN) pipeline for 1D vascular reduced-order modeling.

## 1) Environment Setup (Recommended)

This repository includes `create_venv.sh`, which creates a local virtual environment named `gromenv` and installs the core Python dependencies.

### Step 1: install virtualenv (once)
```bash
python -m pip install virtualenv
```

### Step 2: create the environment from repo root
```bash
bash create_venv.sh
```

### Step 3: activate the environment
```bash
source gromenv/bin/activate
```

After activation, all commands below should be run from the repository root (`VascGraphROM/`).

### What `create_venv.sh` installs
- matplotlib
- vtk
- scipy
- dgl
- torch
- tqdm
- meshio
- pandas
- pyyaml
- pydantic

## 2) Data Path Configuration

1. Download dataset (`gromdata`) from the original project link:  
   https://drive.google.com/open?id=1IByz6kyouNtNgnOxKrFK4DnAVu2yh6S1&authuser=lpego%40stanford.edu&usp=drive_fs
2. Copy `data_location_example.txt` to `data_location.txt`.
3. Edit `data_location.txt` and set it to the **parent directory** that contains your `gromdata` folder.

> Note: `.vtp` files can be visualized with Paraview: https://www.paraview.org

## 3) Training / Evaluation Commands

Use the standardized command format below.

### Train
```bash
python network1d/train.py --config configs/baseline_mgn.yaml
```

### Evaluate (dataset-level)
```bash
python network1d/eval.py --config configs/baseline_mgn.yaml --ckpt models/xxx/trained_gnn.pms
```

### Rollout Evaluation (per-graph)
```bash
python network1d/rollout_eval.py --config configs/baseline_mgn.yaml --ckpt models/xxx/trained_gnn.pms
```

## 4) Compatibility Notes (Common Environment Issues)

### Problem A
```text
ModuleNotFoundError: No module named 'torchdata.datapipes'
```
Fix:
```bash
python -m pip uninstall -y torchdata
python -m pip install --no-deps torchdata==0.7.1
```

### Problem B
```text
Cannot find DGL C++ graphbolt library ...
```
This is usually a PyTorch/DGL version mismatch. A known working combination:

```bash
python -m pip uninstall -y torch torchvision torchaudio dgl numpy
python -m pip install "numpy==1.26.4"
python -m pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install dgl==2.1.0 \
  -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
```

Alternative DGL wheel command:
```bash
python -m pip install "dgl==2.1.0+cu121" -f https://data.dgl.ai/wheels/cu121/repo.html
```

## 5) Project Structure

- `network1d/`: model, training, evaluation, rollout scripts.
- `graph1d/`: graph and dataset generation/normalization utilities.
- `configs/`: YAML experiment configs (`baseline_mgn.yaml` baseline).
- `tools/`: plotting and I/O utilities.
- `test/`: regression tests and sample test assets.

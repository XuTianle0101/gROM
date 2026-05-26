## Graph Reduced Order Models (gROM)

![run_tests](https://github.com/StanfordCBCL/gROM/actions/workflows/run_tests.yml/badge.svg)

gROM provides Graph Neural Network (GNN) workflows for reduced-order cardiovascular simulation in 1D vessel networks.

<p align="center">
  <img src="https://github.com/lucapegolotti/gROM/blob/main/.github/aortofemoral_simulation.gif" alt="Simulation">
</p>

## Repository structure

- `graph1d/`: graph preprocessing, normalization, and dataset generation.
- `network1d/`: model definition, training, rollout, and evaluation.
- `tools/`: utility I/O and plotting helpers.
- `test/`: lightweight regression tests and bundled sample data.

## Setup

### 1) Create environment

```bash
pip install virtualenv
bash create_venv.sh
source gromenv/bin/activate
```

### 2) Configure data path

1. Download data from the project link: https://drive.google.com/open?id=1IByz6kyouNtNgnOxKrFK4DnAVu2yh6S1&authuser=lpego%40stanford.edu&usp=drive_fs
2. Copy `data_location_example.txt` to `data_location.txt`.
3. Edit `data_location.txt` so it points to the folder that contains `gromdata/`.

> Note: `.vtp` files can be inspected with [ParaView](https://www.paraview.org).

## Data generation

You can regenerate graphs from raw data:

```bash
python graph1d/generate_graphs.py
```

## Training

Run training from repository root:

```bash
python network1d/training.py
```

Model outputs are saved to timestamped folders under `models/` and include:
- `trained_gnn.pms`
- `parameters.json`
- history/metric plots

## Evaluation / rollout

Evaluate a trained model:

```bash
python network1d/tester.py models/01.01.1990_00.00.00
```

This computes errors on train/test splits and writes outputs to `results/`.

## Running tests

```bash
bash test/run_test_training.sh
bash test/run_test_rollout.sh
```

## Common environment fixes

### Problem 1

```text
ModuleNotFoundError: No module named 'torchdata.datapipes'
```

Fix:

```bash
python -m pip uninstall -y torchdata
python -m pip install --no-deps torchdata==0.7.1
```

### Problem 2

```text
FileNotFoundError: Cannot find DGL C++ graphbolt library ...
```

Fix (example CUDA 12.1 stack):

```bash
python -m pip uninstall -y torch torchvision torchaudio dgl numpy
python -m pip install "numpy==1.26.4"
python -m pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install dgl==2.1.0 \
  -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
```

Alternative install command:

```bash
pip install "dgl==2.1.0+cu121" -f https://data.dgl.ai/wheels/cu121/repo.html
```

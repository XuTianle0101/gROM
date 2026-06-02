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

### 1) Create environment (includes compatibility fixes)

```bash
pip install virtualenv
bash create_venv.sh
source gromenv/bin/activate
```

`create_venv.sh` installs the tested dependency set from `requirements.txt`
(including `torch==2.2.1`, `torchdata==0.7.1`, `dgl==2.1.0`,
`numpy==1.26.4`) to avoid the known DGL/GraphBolt import failures.

For development without the helper script, install the editable package and
test dependencies directly:

```bash
python -m pip install -e . -r requirements-dev.txt
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

The module entrypoint supports explicit paths and reproducibility options:

```bash
python -m network1d.training --data-location test/test_data/ --graphs-folder graphs/ --epochs 3 --out-dir models/ --seed 10
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

The module entrypoint can also be run with explicit data and result folders:

```bash
python -m network1d.tester test/test_data/gnn_model --data-location test/test_data/ --graphs-folder graphs/ --results-dir results/
```

Add `--plot` to generate rollout videos in `results/`:

```bash
python network1d/tester.py models/01.01.1990_00.00.00 --plot
```

## Running tests

```bash
bash test/run_test_training.sh
bash test/run_test_rollout.sh
```

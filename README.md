## Graph Reduced Order Models ##

![run_tests](https://github.com/StanfordCBCL/gROM/actions/workflows/run_tests.yml/badge.svg)

In this repository we implement reduced order models for cardiovascular simulations using Graph Neural Networks (GNNs).

<p  align="center">
    <img src="https://github.com/lucapegolotti/gROM/blob/main/.github/aortofemoral_simulation.gif" alt="Simulation">
</p>


### Install the virtual environment ###

Let us first install `virtualenv`:

    pip install virtualenv

Then, from the root of the project:

    bash create_venv.sh

This will create a virtual environment `gromenv` with the required dependencies.

### Download the data ###

The data can be downloaded [here](https://drive.google.com/open?id=1IByz6kyouNtNgnOxKrFK4DnAVu2yh6S1&authuser=lpego%40stanford.edu&usp=drive_fs).
Next, duplicate or rename `data_location_example.txt` as `data_location.txt` and set in it the location of the downloaded `gromdata` folder.

Note: `.vtp` files can be  inspected with [Paraview](https://www.paraview.org).

The `gromdata` contains all the data necessary to train the GNN. However, it is possible to regenerate the data by launching `python graph1d/generate_graphs.py` from the root of the project.

### Train a GNN ###

From root, type

    python network1d/training.py

The parameters of the trained model and hyperparameters will be saved in `models`, in a folder named as the date and time when the training was launched.

### Test a GNN ###

Within the directory `graphs`, type

    python network1d/tester.py $NETWORKPATH

For example,

    python network1d/tester.py models/01.01.1990_00.00.00

This compute errors for all train and test geometries.
In the example, `models/01.01.1990_00.00.00` is a model generated after training (see Train a GNN).

Some already-trained models are included in `gromdata`

### Environment Problems ###

Problem1:

```bash
ModuleNotFoundError: No module named 'torchdata.datapipes'
````

The version of torchdata is too high and needs to be downgraded.

```bash
python -m pip uninstall -y torchdata
python -m pip install --no-deps torchdata==0.7.1
```

Problem2:

```bash
FileNotFoundError: Cannot find DGL C++ graphbolt library at /root/autodl-tmp/gROM/gromenv/lib/python3.12/site-packages/dgl/graphbolt/libgraphbolt_pytorch_2.12.0.so
```

The PyTorch version is too high and needs to be downgraded.

```bash
cd /root/autodl-tmp/gROM
source gromenv/bin/activate

python -m pip uninstall -y torch torchvision torchaudio dgl numpy

python -m pip install "numpy==1.26.4"

python -m pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install dgl==2.1.0 \
  -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

```

```bash
pip install "dgl==2.1.0+cu121" -f https://data.dgl.ai/wheels/cu121/repo.html
```
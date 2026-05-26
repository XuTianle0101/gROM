#!/bin/bash
set -euo pipefail

VENVNAME=${VENVNAME:-gromenv}

python3 -m virtualenv "$VENVNAME"
source "$VENVNAME/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Core scientific stack (pin numpy for DGL/Torch compatibility)
python -m pip install "numpy==1.26.4" scipy matplotlib tqdm pandas meshio pyyaml pydantic vtk

# Install a known-compatible PyTorch + TorchData + DGL stack.
# Default: CPU wheels (portable for local/dev/test environments).
python -m pip install "torch==2.2.1" "torchvision==0.17.1" "torchaudio==2.2.1"
python -m pip install --no-deps "torchdata==0.7.1"
python -m pip install "dgl==2.1.0"

echo "Environment created at ./${VENVNAME}"
echo "Activate with: source ${VENVNAME}/bin/activate"

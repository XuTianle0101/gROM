#!/bin/bash
set -euo pipefail

VENVNAME=${VENVNAME:-gromenv}

python3 -m virtualenv "$VENVNAME"
source "$VENVNAME/bin/activate"

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt
python -m pip install -e .

echo "Environment created at ./${VENVNAME}"
echo "Activate with: source ${VENVNAME}/bin/activate"

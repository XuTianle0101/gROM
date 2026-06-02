#!/bin/bash

set -e

source gromenv/bin/activate 
python -m pytest -q test/test_training.py

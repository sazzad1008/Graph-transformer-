# Graph-transformer-

This repository contains reference implementations for heterogeneous graph transformers and vision transformers.
The original experiments are preserved in notebooks, while the reusable components now live in Python modules.

## Project structure

- `graph_transformer/`: Core Python modules.
- `scripts/`: Minimal runnable demos.
- `run.sh`: Convenience script for running the demos.
- `*.ipynb`: Original notebooks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

```bash
bash run.sh
```

You can also run each demo directly:

```bash
python scripts/run_vit.py
python scripts/run_hgt.py
```

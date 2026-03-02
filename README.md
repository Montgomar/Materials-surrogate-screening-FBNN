# Materials Surrogate Modeling & Large-Scale Screening (FBNN)

This repository implements a reproducible workflow for:
1) Training a feed-forward neural network (surrogate model) on experimental materials data,
2) Validating physical constraints (positive predictions),
3) Predicting properties for large candidate sets (e.g., >30k materials),
4) Ranking candidates via multi-objective indices (tribo index & cost index).

## Project Structure
- `src/` training, evaluation, screening, ranking
- `configs/` YAML configs for training and screening
- `data/` data format description and optional demo data (no proprietary data committed)
- `outputs/` run artifacts (checkpoints, plots, metrics)

## Notes
- Original baseline was implemented in MATLAB.
- This repo provides a Python pipeline designed for reproducibility and extensibility.

## Quickstart (Demo)
```bash
pip install -r requirements.txt
python src/train.py --config configs/train.yaml

[//]: # (python src/evaluate.py --run_dir outputs/latest)
python src/predict.py --config configs/screening.yaml
python src/rank.py --config configs/screening.yaml

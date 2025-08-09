# Kaggle Playground Series S5E8  Bank Binary Classification
Reproducible workflow for Kaggle S5E8 (Bank dataset). Predict probability of `y`. Metric: **ROC AUC**.

## Structure
notebooks/  # EDA + baselines
src/        # training pipeline
reports/    # 1-page write-up
docs/images # saved plots
environment.yml / requirements.txt
.github/workflows/python.yml
.gitignore, LICENSE, README.md

## Setup
conda env create -f environment.yml && conda activate s5e8
# or
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Data (dont commit)
pip install kaggle
kaggle competitions download -c playground-series-s5e8
unzip playground-series-s5e8.zip -d data

## Reproduce
jupyter notebook notebooks/eda_and_baseline.ipynb  # (create later)
python src/train.py --data_dir data --model lgbm --n_splits 5 --random_state 42

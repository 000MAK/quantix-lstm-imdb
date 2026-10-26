# LSTM Movie Review Sentiment

Compact word-level LSTM trained on the ACL IMDb review corpus. The official train split is divided into train/validation and official test remains held out. Downloads 84 MB on first run.

## Run

Python 3.10+ recommended. Create a virtual environment and install `pip install -r requirements.txt`. Run `python main.py --help` for options. Data downloads to `data/`; model outputs and metrics are written to `outputs/`. These runtime files are excluded from Git.

This educational baseline uses reproducible splits/seeds and small defaults for local hardware.

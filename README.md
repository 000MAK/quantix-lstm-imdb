# LSTM Sentiment Classification | IMDb Reviews

A compact word-level bidirectional PyTorch LSTM for binary movie-review sentiment classification. The official training corpus is split into train and validation subsets; the official test corpus remains held out.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --epochs 1 --limit 1000
```

The first run downloads the Stanford Large Movie Review Dataset (about 84 MB). Files are stored under `data/`; model weights and metrics are written to `outputs/`. Both are excluded from Git. Run `python main.py --help` for options.

## Model and limits

Tokenization uses a simple lowercase alphabetic vocabulary with a capped sequence length. This is an educational baseline, not a state-of-the-art language model. Results depend on the chosen training subset and epochs.

Dataset source: [Large Movie Review Dataset, Stanford AI](https://ai.stanford.edu/~amaas/data/sentiment/).

## Results

No reference run is published yet. When you run it, report the training-set size, epochs and seed alongside any metric.

## Repository contents

| File | Purpose |
|---|---|
| `.gitignore` | Keeps data, weights and outputs out of Git |
| `LICENSE` | MIT licence for the code |
| `main.py` | Training and evaluation script |
| `requirements.txt` | Python dependencies |

## License

Code is released under the MIT License (see `LICENSE`). Datasets keep their own licences and are downloaded at run time, not redistributed here.

## Related

Part of a set of compact, reproducible ML/DL baselines by [000MAK](https://github.com/000MAK). Flagship research project: [QMASE martial-arts analytics engine](https://github.com/000MAK/quantix-martial-arts-engine).

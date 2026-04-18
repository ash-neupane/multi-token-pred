# data

Input corpus for training. Files here are gitignored.

## Expected files

- `shortjokes.csv` — CSV with a `Joke` column, one joke per row.

## Consumed by

- `data_loader.py` — `ShortJokes` dataset reads `data/shortjokes.csv` (see `data_file` default).
- Downstream: `train.py`, `generate.py`.

## Download

Source: [Short Jokes dataset on Kaggle](https://www.kaggle.com/datasets/abhinavmoudgil95/short-jokes) (`abhinavmoudgil95/short-jokes`).

Using the Kaggle CLI:

```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/ and chmod 600
kaggle datasets download -d abhinavmoudgil95/short-jokes -p data --unzip
```

Or download manually from the Kaggle page and place `shortjokes.csv` in this directory.

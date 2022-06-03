# WideMLP - main code

This code was used to produce the WideMLP results in the paper.

## Get up and running

1. Download the [data folder](https://github.com/drndr/project_ds_textclass/tree/main/multi_label_data) and make sure that the data is placed into a subfolder `./data/multi_label_datasets` in the exact same directory structure or set the DATASET_FOLDER accordingly.

2. Check for static paths such as `CACHE_DIR` in `run_text_classification.py`

3. Check for dependencies `numpy`, `torch`, `tqdm`, `joblib`, `tokenizers`, and `scikit-learn`.


## Code overview

- In `models.py`, you find our implementation for the WideMLP.
- In `data.py`, you find the `load_data()` function which, does the data loading. Valid datasets are: `[ 'amazon', 'dbpedia', 'econbiz', 'nyt', 'pubmed', 'reuters', 'rcv1-v2', 'goemotions']

## Running experiments

The script run\_text\_classification.py is the main entry point for running an experiment.
Within the experiments folder, you find the bash scripts that we used for the experiments.
The results_mlp.csv contains the experiment results.

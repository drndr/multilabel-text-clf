# WideMLP - code for Pubmed experiments

This code was used to experiment with the Pubmed dataset. Pubmed contains over 12 million documents and 27,773 classes. Because of Pubmeds size, we ran into memory-problems for generating the document-label matrix. To solve this problem, we implemented the same setup as in multilabel-processing, but generated a sparse matrix instead of a dense matrix. Other models ran into similar problems, thats why we dropped this dataset from the paper. The code for experimenting with Pubmed is just included for completeness and further experiments.

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

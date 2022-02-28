# Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP -- Code for Experiments

## Get up and running

1. Download the data folder from serverc and make sure that the data is placed into a subfolder `./data/multi_label_datasets` in the exact same directory structure or set the DATASET_FOLDER accordingly.

2. Check for static paths such as `CACHE_DIR` in `run_text_classification.py`

3. Check for dependencies `numpy`, `torch`, `transformers`, `tqdm`, `joblib`, `tokenizers`, and `scikit-learn`.


## Code overview

- In `models.py`, you find our implementation for the WideMLP.
- In `data.py`, you find the `load_data()` function which, does the data loading. Valid datasets are: `[ 'amazon', 'dbpedia', 'econbiz', 'nyt', 'pubmed', 'reuters', 'rcv1-v2', 'goemotions']
- To run experiments on pubmed check out the sparse-multilabel-processing

## Running experiments

The script run\_text\_classification.py is the main entry point for running an experiment.
Within the experiments folder, you find the bash scripts that we used for the experiments.
The results_mlp.csv contains the experiment results.
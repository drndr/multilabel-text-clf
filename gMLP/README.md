This folder contains our implementation of the gMLP/aMLP model based on the paper [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050) for single-label and multi-label text classification.

## Get up and running

1. Make sure that the datasets are placed into a subfolders `./data/multi_label_data` and `./data/single_label_datasets`.

2. Check for dependencies `numpy`, `torch`, `transformers`, `pandas`, `nltk`, `sklearn`, `einops` and `matplotlib`


## Code overview

- In `train_test_single_label.py`, you find our implementation for the gMLP/aMLP models for single label classification.
  The experimental setup can be configured in the begining of the script:
    - valid datasets for the 'dataset' variable and corresponding label number : { '20ng' - 20 , 'R8' - 8 , 'R52' - 52 , 'ohsumed' - 23 , 'mr' 2 }
    - has_attention variable sets the model type: True = aMLP, False = gMLP  
- In `train_test_multi_label.py` you find our implementation for the gMLP/aMLP models for multi label classification.
  The experimental setup can be configured in the begining of the script:
    - valid datasets for the 'dataset' variable and corresponding label number: { 'amazon' - 531, 'dbpedia' - 298, 'econbiz' - 5661, 'nyt' - 166, 'reuters' - 90, 'rcv1-v2' - 103, 'goemotions' -28 }
    - has_attention variable sets the model type: True = aMLP, False = gMLP
- In `test_multi_label.py` you find a testing script for saved multi-label models. 
    - pretrained_model variable defines the path to the saved model
    - valid dataset, label number and model type is defined the same way as in 'train_test_multilabel'

## Running experiments

The scripts `train_test_multi_label.py` and `train_test_single_label.py` are the main entry point for running an experiment. Variables at the begining of the scripts should be set accordingly to datasets and model type.
